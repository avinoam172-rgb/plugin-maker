// Plugin Maker - builder (version 1.3)
// Runs inside GitHub Actions: asks a free AI model for the code of a plugin or a mod,
// compiles it (Maven for plugins, Gradle for Forge / Fabric mods), sends compile
// errors back to the AI for fixing, and writes the result into ./out
// (JAR + source zip + result.json).
// No dependencies - plain Node.js.

import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = process.cwd();
const OUT = path.join(ROOT, 'out');
const WORK = path.join(ROOT, 'work');
const env = process.env;

const MC_RX = /^(1\.21(\.\d{1,2})?|2[6-9]\.\d{1,2}(\.\d{1,2})?)$/;
const MAX_ATTEMPTS = 3;
const MAX_PROMPT = 1500;

// Fabric build tools that were tested together
const FABRIC_OLD = { loom: '1.16.3', plugin: 'fabric-loom' };              // Minecraft 1.21.x
const FABRIC_NEW = { loom: '1.19.4', plugin: 'net.fabricmc.fabric-loom' }; // Minecraft 26.x and later

const SPIGOT_META = 'https://hub.spigotmc.org/nexus/content/repositories/snapshots/org/spigotmc/spigot-api/maven-metadata.xml';
const FORGE_MAVEN = 'https://maven.minecraftforge.net/net/minecraftforge/forge/';
const FABRIC_MAVEN = 'https://maven.fabricmc.net/net/fabricmc/';

const PROVIDERS = {
  gemini: {
    url: 'https://generativelanguage.googleapis.com/v1beta/openai/chat/completions',
    key: env.GEMINI_API_KEY, maxOut: 16000, budget: 0,
  },
  groq: {
    url: 'https://api.groq.com/openai/v1/chat/completions',
    key: env.GROQ_API_KEY, maxOut: 5000, budget: 7600, // free plan: 8K tokens per minute
  },
  openrouter: {
    url: 'https://openrouter.ai/api/v1/chat/completions',
    key: env.OPENROUTER_API_KEY, maxOut: 8000, budget: 0,
  },
};
const DEFAULT_CHAIN =
  'gemini:gemini-flash-latest,groq:openai/gpt-oss-120b,groq:qwen/qwen3.8-27b,' +
  'gemini:gemini-flash-lite-latest,groq:openai/gpt-oss-20b,openrouter:openrouter/free';

const FORBIDDEN = [
  [/Runtime\s*\.\s*getRuntime/, 'Runtime.getRuntime'],
  [/ProcessBuilder/, 'ProcessBuilder'],
  [/\bjava\.net\./, 'java.net (network access)'],
  [/\bjavax\.net\./, 'javax.net (network access)'],
  [/URLClassLoader|defineClass/, 'class loading'],
  [/\bjavax\.script\b/, 'javax.script'],
  [/\bsun\.misc\b/, 'sun.misc'],
  [/\bjava\.lang\.reflect\b|Class\s*\.\s*forName/, 'reflection'],
  [/System\s*\.\s*exit/, 'System.exit'],
  [/\bjava\.nio\.channels\b|\bjava\.rmi\b|\bjavax\.naming\b/, 'low-level IO / RMI / JNDI'],
];

// ---------------------------------------------------------------- helpers

function sanitizeName(s) {
  const n = String(s || '').replace(/[^A-Za-z0-9_]/g, '');
  return /^[A-Za-z][A-Za-z0-9_]{2,31}$/.test(n) ? n : '';
}

function finish(result) {
  fs.mkdirSync(OUT, { recursive: true });
  fs.writeFileSync(path.join(OUT, 'result.json'), JSON.stringify(result, null, 2));
  console.log('RESULT:', JSON.stringify({ ...result, detail: undefined, explain: undefined }));
  process.exit(0);
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const isNewMc = (mc) => !mc.startsWith('1.'); // 26.x and later need Java 25

function javaHome(version) {
  return env['JAVA_HOME_' + version + '_X64'] || env.JAVA_HOME || '';
}

// rough token estimate: Hebrew and other non-ASCII letters cost about a token each
function estTokens(messages) {
  let t = 0;
  for (const m of messages) {
    const s = m.content;
    const non = (s.match(/[^\x00-\x7f]/g) || []).length;
    t += non + Math.ceil((s.length - non) / 3) + 8;
  }
  return t;
}

function cleanEnv(extra) {
  const out = {};
  for (const [k, v] of Object.entries(env)) {
    if (/KEY|TOKEN|SECRET|PASSWORD|PROMPT/i.test(k)) continue;
    out[k] = v;
  }
  return Object.assign(out, extra || {});
}

function download(url, outFile) {
  const args = ['-fsSL', '--retry', '3', '--retry-delay', '3', '--max-time', '180', url];
  if (outFile) args.push('-o', outFile);
  const r = spawnSync('curl', args, { encoding: outFile ? undefined : 'utf8', env: cleanEnv(), maxBuffer: 64 * 1024 * 1024 });
  if (r.status !== 0) return null;
  return outFile ? true : r.stdout;
}

function compareVersions(a, b) {
  const pa = a.split(/[^0-9]+/).filter(Boolean).map(Number);
  const pb = b.split(/[^0-9]+/).filter(Boolean).map(Number);
  for (let i = 0; i < Math.max(pa.length, pb.length); i++) {
    const d = (pa[i] || 0) - (pb[i] || 0);
    if (d) return d;
  }
  return 0;
}

function mavenVersions(metaUrl) {
  const xml = download(metaUrl);
  if (!xml) return null;
  return [...xml.matchAll(/<version>([^<]+)<\/version>/g)].map((m) => m[1]);
}

// ---------------------------------------------------------------- AI

function buildChain() {
  const raw = (env.AI_CHAIN || '').trim() || DEFAULT_CHAIN;
  const chain = [];
  for (const item of raw.split(',')) {
    const i = item.indexOf(':');
    if (i < 1) continue;
    const prov = item.slice(0, i).trim();
    const model = item.slice(i + 1).trim();
    const p = PROVIDERS[prov];
    if (!p || !p.key || !model) continue;
    chain.push({ prov, model, dead: false });
  }
  return chain;
}

async function callOnce(entry, messages, withExtras) {
  const p = PROVIDERS[entry.prov];
  let maxTokens = p.maxOut;
  if (p.budget) {
    maxTokens = Math.min(p.maxOut, p.budget - estTokens(messages));
    if (maxTokens < 1500) throw new Error('request too large for this model');
  }
  const body = { model: entry.model, messages, temperature: 0.2, max_tokens: maxTokens };
  if (withExtras && /gpt-oss/.test(entry.model)) body.reasoning_effort = 'low';
  const res = await fetch(env.AI_TEST_URL || p.url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Authorization: 'Bearer ' + p.key },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(150000),
  });
  const text = await res.text();
  if (!res.ok) {
    const err = new Error('HTTP ' + res.status + ' ' + text.slice(0, 200).replace(/\s+/g, ' '));
    err.status = res.status;
    err.retryAfter = parseFloat(res.headers.get('retry-after') || '0') || 0;
    throw err;
  }
  let data;
  try { data = JSON.parse(text); } catch { throw new Error('bad JSON from model'); }
  const choice = data.choices && data.choices[0];
  const content = choice && choice.message && choice.message.content;
  if (!content || typeof content !== 'string') throw new Error('empty answer');
  if (choice.finish_reason === 'length') throw new Error('answer was cut off (too long)');
  return content;
}

async function callEntry(entry, messages) {
  let extras = true;
  let waited = false;
  let netRetries = 0;
  for (let i = 0; i < 6; i++) {
    try {
      return await callOnce(entry, messages, extras);
    } catch (e) {
      if (e.status === 400 && extras) { extras = false; continue; }
      if (!e.status && e.name !== 'TimeoutError' && !/too large|cut off|empty answer|bad JSON/.test(e.message) && netRetries < 2) {
        netRetries++;
        console.log('  network error, retrying: ' + (e.cause && e.cause.code || e.message));
        await sleep(2000);
        continue;
      }
      if (e.status === 429 && !waited && e.retryAfter > 0 && e.retryAfter <= 45) {
        waited = true;
        console.log(`  rate limited, waiting ${e.retryAfter}s`);
        await sleep((e.retryAfter + 1) * 1000);
        continue;
      }
      throw e;
    }
  }
  throw new Error('gave up');
}

// tries the models in order; a model that fails is skipped for the rest of the run
async function askAI(chain, messages) {
  for (const entry of chain) {
    if (entry.dead) continue;
    const label = entry.prov + ':' + entry.model;
    try {
      console.log('asking ' + label);
      const out = await callEntry(entry, messages);
      return { text: out, model: label };
    } catch (e) {
      console.log('  failed: ' + e.message);
      entry.dead = true;
    }
  }
  return null;
}

const FABRIC_EXAMPLE = `package me.aimod.example;

import net.fabricmc.api.ModInitializer;
import net.fabricmc.fabric.api.command.v2.CommandRegistrationCallback;
import net.minecraft.commands.Commands;
import net.minecraft.network.chat.Component;
import net.minecraft.server.level.ServerPlayer;

public class ExampleMod implements ModInitializer {
    @Override
    public void onInitialize() {
        CommandRegistrationCallback.EVENT.register((dispatcher, registryAccess, environment) ->
            dispatcher.register(Commands.literal("heal").executes(ctx -> {
                ServerPlayer player = ctx.getSource().getPlayerOrException();
                player.setHealth(player.getMaxHealth());
                ctx.getSource().sendSuccess(() -> Component.literal("Healed!"), false);
                return 1;
            })));
    }
}`;

function systemPrompt(kind, mc, wantedName, example) {
  const java = isNewMc(mc) ? 25 : 21;
  const lines = [];
  const format = [
    'If the request is not a request for a Minecraft ' + (kind === 'plugin' ? 'plugin' : 'mod') +
      ', or asks for something from the forbidden list, answer with a single line:',
    'REFUSE: <short reason, in the language of the request>',
    '',
    'Otherwise answer in EXACTLY this format and nothing else (no markdown code fences):',
    '',
  ];
  const common = [
    '- Keep it compact: 1 to 4 Java files, at most about 250 lines in total. No placeholder or unfinished code.',
    '- Use only API that really exists in this exact version. When unsure, use the simplest well-known API.',
    '- Text shown to players is written in the language of the request.',
    '- Never use: network access (java.net), Runtime.exec, ProcessBuilder, reflection, class loading, System.exit, or files outside the game/plugin data folders.',
    '- Never write hidden behaviour, backdoors, secret operator access, or anything meant to damage a server, a world or a computer, or to steal data.',
    wantedName
      ? '- The name is exactly: ' + wantedName
      : '- Pick a short English name (letters and digits only, starting with a letter).',
  ];
  const explain = [
    '=== EXPLAIN ===',
    '<2 to 6 short lines in the language of the request: what it does, the commands or how to use it, and anything you changed, simplified or left out compared to the request>',
    '=== END ===',
  ];

  if (kind === 'plugin') {
    lines.push(
      'You write Minecraft server plugins. You receive a request (often in Hebrew) and answer with the complete source of one small plugin.',
      '',
      'Target: Minecraft ' + mc + ', Spigot API (package org.bukkit) so it runs on Spigot, Paper and Purpur. Java ' + java + '.',
      '',
      'Rules:',
      '- Use ONLY the Bukkit/Spigot API and the Java standard library. No Paper-only classes, no Adventure/MiniMessage, no NMS, no CraftBukkit classes, no external libraries.',
      '- The main class extends org.bukkit.plugin.java.JavaPlugin.',
      '- Every command is declared in plugin.yml and handled in code. Register every listener.',
      '- Use org.bukkit.ChatColor for colors.',
      ...common, '', ...format,
      'NAME: <name>',
      '=== FILE: src/main/resources/plugin.yml ===',
      '<file content>',
      '=== FILE: src/main/java/<package path>/<ClassName>.java ===',
      '<file content>',
      ...explain,
      '',
      'plugin.yml must contain name, version, main, api-version and the commands section when there are commands.',
    );
    return lines.join('\n');
  }

  const forge = kind === 'forge';
  lines.push(
    'You write Minecraft mods. You receive a request (often in Hebrew) and answer with the complete Java source of one small mod.',
    '',
    'Target: Minecraft ' + mc + ', ' + (forge ? 'Minecraft Forge (package net.minecraftforge, NOT NeoForge)' : 'Fabric Loader with Fabric API') +
      ', official Mojang mappings (names like net.minecraft.server.level.ServerPlayer, net.minecraft.network.chat.Component). Java ' + java + '.',
    '',
    'Rules:',
    forge
      ? '- The main class is annotated with @Mod("<mod id>") where the mod id is the NAME in lowercase.'
      : '- The main class implements net.fabricmc.api.ModInitializer. A client-only class (optional) implements net.fabricmc.api.ClientModInitializer.',
    '- Do NOT write build files, ' + (forge ? 'mods.toml' : 'fabric.mod.json') + ' or pack.mcmeta - they are generated for you.',
    '- No mixins, no access transformers / access wideners, no external libraries.',
    '- Prefer features that need no new textures or models (commands, events, game mechanics). You cannot provide image files; a new item or block would show a missing texture.',
    '- The only resource files allowed are JSON files under src/main/resources/assets/<mod id>/ or src/main/resources/data/<mod id>/ (for example lang files).',
    ...common, '', ...format,
    'NAME: <name>',
    'MAIN_CLASS: <fully qualified main class>',
  );
  if (!forge) lines.push('CLIENT_CLASS: <fully qualified client class, or the word none>');
  lines.push(
    '=== FILE: src/main/java/<package path>/<ClassName>.java ===',
    '<file content>',
    ...explain,
  );
  if (example) {
    lines.push('', 'Reference code that compiles for this exact version (use it to see the correct imports and API style, do not copy its features):', example);
  }
  return lines.join('\n');
}

// ---------------------------------------------------------------- parsing and checks

const JAVA_PATH = /^src\/main\/java\/(?:[A-Za-z_][A-Za-z0-9_]*\/)*[A-Za-z_][A-Za-z0-9_]*\.java$/;
const PLUGIN_RES = /^src\/main\/resources\/(?:[A-Za-z0-9_-]+\/)*[A-Za-z0-9_-]+\.(?:yml|yaml|txt|json|properties)$/;
const MOD_RES = /^src\/main\/resources\/(?:assets|data)\/[a-z0-9_]+\/(?:[a-z0-9_.-]+\/)*[a-z0-9_.-]+\.json$/;
const CLASS_NAME = /^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+$/;

function parseAnswer(text) {
  const t = text.replace(/\r\n/g, '\n').trim();
  const refuse = t.match(/^REFUSE:\s*(.+)$/m);
  if (refuse && !/=== FILE:/.test(t)) return { refuse: refuse[1].trim().slice(0, 300) };

  const head = (key) => {
    const m = t.match(new RegExp('^' + key + ':\\s*(\\S+)', 'm'));
    return m ? m[1] : '';
  };
  const files = new Map();
  const re = /^=== (FILE:\s*(.+?)|EXPLAIN|END) ===[ \t]*$/gm;
  const marks = [];
  let m;
  while ((m = re.exec(t)) !== null) marks.push({ kind: m[1].split(':')[0], p: m[2] || '', start: m.index, bodyStart: re.lastIndex });
  let explain = '';
  for (let i = 0; i < marks.length; i++) {
    if (marks[i].kind === 'END') continue;
    let body = t.slice(marks[i].bodyStart, i + 1 < marks.length ? marks[i + 1].start : t.length);
    body = body.replace(/^\s*\n/, '').replace(/\s+$/, '');
    body = body.replace(/^```[A-Za-z]*\n/, '').replace(/\n```$/, '');
    if (marks[i].kind === 'EXPLAIN') { explain = body.trim().slice(0, 1200); continue; }
    const p = marks[i].p.replace(/\\/g, '/').replace(/^\.?\//, '');
    files.set(p, body + '\n');
  }
  return {
    name: head('NAME') || head('PLUGIN_NAME') || head('MOD_NAME'),
    mainClass: head('MAIN_CLASS'),
    clientClass: head('CLIENT_CLASS'),
    files, explain,
  };
}

// returns a list of problems (empty = fine)
function checkFiles(files, kind) {
  const problems = [];
  if (files.size === 0) return ['The answer contained no "=== FILE: ... ===" blocks. Use the exact output format.'];
  if (files.size > 12) problems.push('Too many files (max 12).');
  const resOk = kind === 'plugin' ? PLUGIN_RES : MOD_RES;
  let total = 0;
  let javaCount = 0;
  for (const [p, body] of files) {
    total += body.length;
    if (p.includes('..') || /META-INF/i.test(p) || !(JAVA_PATH.test(p) || resOk.test(p))) {
      problems.push('File path not allowed: ' + p.slice(0, 120) + (kind === 'plugin'
        ? ' (only src/main/java/**.java and src/main/resources/*.yml|txt|json|properties)'
        : ' (only src/main/java/**.java and JSON files under src/main/resources/assets/<mod id>/ or data/<mod id>/). Do not write build files, mods.toml, fabric.mod.json or pack.mcmeta.'));
      continue;
    }
    if (body.length > 60000) problems.push('File too large: ' + p);
    if (p.endsWith('.java')) {
      javaCount++;
      for (const [rx, label] of FORBIDDEN) {
        if (rx.test(body)) problems.push('Forbidden API in ' + p + ': ' + label + '. Rewrite without it.');
      }
    }
  }
  if (total > 200000) problems.push('Total size too large.');
  if (javaCount === 0) problems.push('No Java source file.');
  if (kind === 'plugin' && !files.has('src/main/resources/plugin.yml')) problems.push('Missing src/main/resources/plugin.yml.');
  return problems;
}

// forces the right name and api-version, and checks that "main" points to a real class
function fixPluginYml(files, name, mc) {
  const key = 'src/main/resources/plugin.yml';
  let yml = files.get(key);
  const main = yml.match(/^main:\s*['"]?([A-Za-z0-9_.]+)['"]?\s*$/m);
  if (!main) return 'plugin.yml has no valid "main:" line.';
  const mainFile = 'src/main/java/' + main[1].replace(/\./g, '/') + '.java';
  if (!files.has(mainFile)) {
    return 'plugin.yml main is "' + main[1] + '" but there is no file ' + mainFile + '.';
  }
  const api = mc.split('.').slice(0, 2).join('.');
  const setLine = (k, v) => {
    const rx = new RegExp('^' + k + ':.*$', 'm');
    yml = rx.test(yml) ? yml.replace(rx, k + ': ' + v) : k + ': ' + v + '\n' + yml;
  };
  setLine('api-version', "'" + api + "'");
  if (!/^version:/m.test(yml)) setLine('version', "'1.0'");
  setLine('name', name);
  files.set(key, yml);
  return '';
}

function classFile(cls) {
  return 'src/main/java/' + cls.replace(/\./g, '/') + '.java';
}

function checkMod(parsed, kind, modId) {
  const problems = [];
  const files = parsed.files;
  if (!CLASS_NAME.test(parsed.mainClass) || !files.has(classFile(parsed.mainClass))) {
    problems.push('The MAIN_CLASS line must name the fully qualified main class, and that file must be included. Got: "' + parsed.mainClass + '".');
  }
  if (kind === 'fabric') {
    const c = parsed.clientClass;
    if (c && c.toLowerCase() !== 'none' && (!CLASS_NAME.test(c) || !files.has(classFile(c)))) {
      problems.push('CLIENT_CLASS must be "none" or a fully qualified class whose file is included. Got: "' + c + '".');
    }
  } else {
    let ok = false;
    for (const [p, body] of files) if (p.endsWith('.java') && /@Mod\s*\(/.test(body) && body.includes('"' + modId + '"')) ok = true;
    if (!ok) problems.push('The main class must be annotated with @Mod("' + modId + '") - use exactly this mod id string.');
  }
  for (const p of files.keys()) {
    const m = p.match(/^src\/main\/resources\/(?:assets|data)\/([a-z0-9_]+)\//);
    if (m && m[1] !== modId && m[1] !== 'minecraft') problems.push('Resource folder must use the mod id "' + modId + '": ' + p);
  }
  return problems;
}

// ---------------------------------------------------------------- project setup per kind

function writeFiles(files) {
  for (const [p, body] of files) {
    const full = path.join(WORK, p);
    if (!full.startsWith(WORK + path.sep)) throw new Error('bad path');
    fs.mkdirSync(path.dirname(full), { recursive: true });
    fs.writeFileSync(full, body);
  }
}

function resetWork() {
  fs.rmSync(WORK, { recursive: true, force: true });
  fs.mkdirSync(WORK, { recursive: true });
}

// --- plugin (Maven)

function preparePlugin(mc) {
  const versions = mavenVersions(SPIGOT_META);
  if (versions && !versions.includes(mc + '-R0.1-SNAPSHOT')) {
    finish({ ok: false, error: 'אין עדיין Spigot API לגרסה ' + mc + '. בחרו גרסה אחרת.' });
  }
  return { java: isNewMc(mc) ? 25 : 21 };
}

function writePlugin(ctx, files, name, mc) {
  resetWork();
  let pom = fs.readFileSync(path.join(HERE, 'pom.xml'), 'utf8');
  pom = pom.replace('<plugin.name>Plugin</plugin.name>', '<plugin.name>' + name + '</plugin.name>')
           .replace('<mc.version>1.21.11</mc.version>', '<mc.version>' + mc + '</mc.version>')
           .replace('<java.release>21</java.release>', '<java.release>' + ctx.java + '</java.release>');
  fs.writeFileSync(path.join(WORK, 'pom.xml'), pom);
  writeFiles(files);
}

// --- Fabric (Gradle + Loom)

function prepareFabric(mc) {
  const api = mavenVersions(FABRIC_MAVEN + 'fabric-api/fabric-api/maven-metadata.xml');
  const loaders = mavenVersions(FABRIC_MAVEN + 'fabric-loader/maven-metadata.xml');
  if (!api || !loaders) finish({ ok: false, error: 'תקלה זמנית בגישה לשרתי Fabric. נסו שוב בעוד כמה דקות.' });
  const mine = api.filter((v) => v.endsWith('+' + mc)).sort(compareVersions);
  if (mine.length === 0) finish({ ok: false, error: 'אין עדיין Fabric API לגרסה ' + mc + '. בחרו גרסה אחרת.' });
  const stable = loaders.filter((v) => /^\d+\.\d+\.\d+$/.test(v)).sort(compareVersions);
  const tools = isNewMc(mc) ? FABRIC_NEW : FABRIC_OLD;
  return {
    java: isNewMc(mc) ? 25 : 21, tools, fabricApi: mine[mine.length - 1], loader: stable[stable.length - 1],
    example: FABRIC_EXAMPLE,
  };
}

function writeFabric(ctx, parsed, name, modId, mc) {
  resetWork();
  const dep = isNewMc(mc) ? 'implementation' : 'modImplementation';
  fs.writeFileSync(path.join(WORK, 'settings.gradle'), [
    'pluginManagement {',
    '    repositories {',
    "        maven { url = 'https://maven.fabricmc.net/' }",
    '        mavenCentral()',
    '        gradlePluginPortal()',
    '    }',
    '}',
    "rootProject.name = '" + modId + "'",
    '',
  ].join('\n'));
  fs.writeFileSync(path.join(WORK, 'build.gradle'), [
    'plugins {',
    "    id '" + ctx.tools.plugin + "' version '" + ctx.tools.loom + "'",
    '}',
    "version = '1.0'",
    "group = 'me.aimod'",
    "base { archivesName = '" + modId + "' }",
    'repositories { mavenCentral() }',
    'dependencies {',
    '    minecraft "com.mojang:minecraft:' + mc + '"',
    isNewMc(mc) ? '' : '    mappings loom.officialMojangMappings()',
    '    ' + dep + ' "net.fabricmc:fabric-loader:' + ctx.loader + '"',
    '    ' + dep + ' "net.fabricmc.fabric-api:fabric-api:' + ctx.fabricApi + '"',
    '}',
    'tasks.withType(JavaCompile).configureEach {',
    '    it.options.release = ' + ctx.java,
    "    it.options.encoding = 'UTF-8'",
    "    it.options.compilerArgs << '-proc:none'",
    '}',
    '',
  ].filter((l) => l !== '').join('\n') + '\n');
  fs.writeFileSync(path.join(WORK, 'gradle.properties'), 'org.gradle.jvmargs=-Xmx3G\norg.gradle.daemon=false\n' + toolchainLine());
  const entry = { main: [parsed.mainClass] };
  const c = parsed.clientClass;
  if (c && c.toLowerCase() !== 'none') entry.client = [c];
  const modJson = {
    schemaVersion: 1, id: modId, version: '1.0', name, description: 'Made with Plugin Maker', environment: '*',
    entrypoints: entry,
    depends: { fabricloader: '>=0.16.0', minecraft: mc, java: '>=' + ctx.java, 'fabric-api': '*' },
  };
  writeFiles(parsed.files);
  fs.mkdirSync(path.join(WORK, 'src/main/resources'), { recursive: true });
  fs.writeFileSync(path.join(WORK, 'src/main/resources/fabric.mod.json'), JSON.stringify(modJson, null, 2) + '\n');
}

// --- Forge (the official MDK of the exact version is the build template)

function toolchainLine() {
  const homes = [javaHome(21), javaHome(25)].filter((h, i, a) => h && a.indexOf(h) === i);
  return homes.length ? 'org.gradle.java.installations.paths=' + homes.join(',') + '\n' : '';
}

function listJava(dir, out = []) {
  if (!fs.existsSync(dir)) return out;
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, e.name);
    if (e.isDirectory()) listJava(full, out);
    else if (e.name.endsWith('.java')) out.push(full);
  }
  return out;
}

function prepareForge(mc) {
  const all = mavenVersions(FORGE_MAVEN + 'maven-metadata.xml');
  if (!all) finish({ ok: false, error: 'תקלה זמנית בגישה לשרתי Forge. נסו שוב בעוד כמה דקות.' });
  const mine = all.filter((v) => v.startsWith(mc + '-')).sort(compareVersions);
  if (mine.length === 0) finish({ ok: false, error: 'אין Forge לגרסה ' + mc + '. בחרו גרסה אחרת.' });
  const full = mine[mine.length - 1];
  const mdk = path.join(ROOT, 'mdk');
  fs.rmSync(mdk, { recursive: true, force: true });
  fs.mkdirSync(mdk, { recursive: true });
  const zip = path.join(mdk, 'mdk.zip');
  if (!download(FORGE_MAVEN + full + '/forge-' + full + '-mdk.zip', zip)) {
    finish({ ok: false, error: 'לא הצלחנו להוריד את ערכת הפיתוח של Forge ' + full + '. נסו שוב בעוד כמה דקות.' });
  }
  const un = spawnSync('unzip', ['-q', '-o', zip, '-d', mdk]);
  if (un.status !== 0) finish({ ok: false, error: 'ערכת הפיתוח של Forge פגומה. נסו שוב.' });
  fs.rmSync(zip);

  // the MDK example sources teach the AI the API style of this exact Forge version
  let example = '';
  const strip = (f) => fs.readFileSync(f, 'utf8')
    .replace(/\/\*[\s\S]*?\*\//g, '').replace(/^\s*\/\/.*$/gm, '').replace(/[ \t]+\/\/ .*$/gm, '')
    .replace(/^[ \t]+/gm, (m) => ' '.repeat(Math.floor(m.length / 2))).replace(/\n\s*\n+/g, '\n');
  const sources = listJava(path.join(mdk, 'src/main/java')).map(strip);
  // the main mod class (the one with @Mod) is the useful one; add the others only if there is room
  sources.sort((x, y) => (/@Mod\s*\(/.test(y) ? 1 : 0) - (/@Mod\s*\(/.test(x) ? 1 : 0));
  for (const src of sources) if (!example || example.length + src.length <= 6500) example += src + '\n';
  const wrapper = fs.readFileSync(path.join(mdk, 'gradle/wrapper/gradle-wrapper.properties'), 'utf8');
  const gm = wrapper.match(/gradle-(\d+)\.(\d+)/);
  const gradleMajor = gm ? parseInt(gm[1], 10) : 9;
  return {
    java: isNewMc(mc) ? 25 : 21, forge: full, mdk, example: example.trim().slice(0, 6500),
    launcherJava: gradleMajor >= 9 ? 25 : 21,
  };
}

function writeForge(ctx, parsed, name, modId, mc) {
  resetWork();
  fs.cpSync(ctx.mdk, WORK, { recursive: true });
  fs.rmSync(path.join(WORK, 'src/main/java'), { recursive: true, force: true });
  fs.rmSync(path.join(WORK, 'src/generated'), { recursive: true, force: true });
  fs.rmSync(path.join(WORK, 'src/test'), { recursive: true, force: true });

  const edit = (rel, fn) => {
    const f = path.join(WORK, rel);
    if (fs.existsSync(f)) fs.writeFileSync(f, fn(fs.readFileSync(f, 'utf8')));
  };
  const rename = (s) => s.split('examplemod').join(modId).split('Example Mod').join(name);

  edit('settings.gradle', (s) =>
    'pluginManagement {\n    repositories {\n        maven { url = \'https://maven.minecraftforge.net/\' }\n' +
    '        mavenCentral()\n        gradlePluginPortal()\n    }\n}\n' +
    rename(s.replace(/pluginManagement\s*\{[\s\S]*?\n\}\s*/, '').replace(/plugins\s*\{[^}]*foojay[^}]*\}\s*/, '')));
  edit('build.gradle', (s) => rename(s).replace(/^version\s*=\s*'[^']*'/m, "version = '1.0'")
    + "\ntasks.withType(JavaCompile).configureEach { options.compilerArgs << '-proc:none' }\n");
  const usesExpand = /expand/.test(fs.readFileSync(path.join(WORK, 'build.gradle'), 'utf8'));
  edit('gradle.properties', (s) => {
    let g = rename(s)
      .replace(/^org\.gradle\.configuration-cache=.*$/m, 'org.gradle.configuration-cache=false')
      .replace(/^org\.gradle\.parallel=.*$/m, 'org.gradle.parallel=false')
      .replace(/^org\.gradle\.daemon=.*$/m, '');
    const set = (k, v) => { if (new RegExp('^' + k + '=', 'm').test(g)) g = g.replace(new RegExp('^' + k + '=.*$', 'm'), k + '=' + v); };
    set('mod_id', modId); set('mod_name', name); set('mod_version', '1.0'); set('mod_group_id', 'me.aimod');
    set('mod_authors', 'Plugin Maker'); set('mod_description', 'Made with Plugin Maker');
    return g + '\norg.gradle.daemon=false\n' + toolchainLine();
  });
  const fixRes = (s) => {
    let r = rename(s);
    if (!usesExpand) r = r.split('${mod_id}').join(modId).split('${mod_name}').join(name);
    return r;
  };
  edit('src/main/resources/META-INF/mods.toml', (t) => fixRes(t)
    .replace(/^authors\s*=.*$/m, 'authors="Plugin Maker"')
    .replace(/^description\s*=\s*'''[\s\S]*?'''/m, "description='''Made with Plugin Maker'''"));
  edit('src/main/resources/pack.mcmeta', fixRes);
  writeFiles(parsed.files);
  try { fs.chmodSync(path.join(WORK, 'gradlew'), 0o755); } catch { /* ignore */ }
}

// ---------------------------------------------------------------- compile

function mavenErrors(all) {
  const seen = new Set();
  const lines = [];
  for (let line of all.split('\n')) {
    if (!line.includes('[ERROR]')) continue;
    line = line.split(WORK + '/').join('').replace('[ERROR]', '').trim();
    if (!line || seen.has(line)) continue;
    if (/^(->|To see the full|Re-run Maven|For more information|\[Help)/.test(line)) continue;
    seen.add(line);
    lines.push(line);
    if (lines.length >= 25) break;
  }
  return lines.join('\n').slice(0, 3000);
}

function compilePlugin(ctx) {
  const r = spawnSync('mvn', ['-B', '-q', '-ntp', 'package', '-DskipTests'], {
    cwd: WORK, env: cleanEnv(javaHome(ctx.java) ? { JAVA_HOME: javaHome(ctx.java) } : {}),
    encoding: 'utf8', timeout: 8 * 60 * 1000, maxBuffer: 32 * 1024 * 1024,
  });
  if (r.error) return { ok: false, errors: 'Maven could not run: ' + r.error.message, infra: true };
  if (r.status === 0) return { ok: true };
  const all = (r.stdout || '') + '\n' + (r.stderr || '');
  const errors = mavenErrors(all) || all.slice(-1500);
  const infra = !/\.java/.test(errors) && /resolve|transfer|Could not find artifact|Connection|timed out/i.test(errors);
  return { ok: false, errors, infra };
}

function gradleErrors(all) {
  const src = all.split(WORK + '/').join('').split('\n');
  const out = [];
  let count = 0;
  for (let i = 0; i < src.length && count < 20; i++) {
    if (/\.java:\d+: error:/.test(src[i])) {
      count++;
      out.push(src[i].trim());
      for (let j = 1; j <= 3 && i + j < src.length; j++) {
        const next = src[i + j];
        if (/\.java:\d+: (error|warning):/.test(next) || /^\d+ errors?$/.test(next.trim())) break;
        if (next.trim()) out.push('  ' + next.trimEnd());
      }
    }
  }
  return out.join('\n').slice(0, 3000);
}

function compileGradle(ctx, kind) {
  const launcher = kind === 'forge' ? ctx.launcherJava : ctx.java;
  const cmd = kind === 'forge' ? './gradlew' : 'gradle';
  const args = ['build', '--no-daemon', '--console=plain'].concat((env.PM_GRADLE_ARGS || '').split(' ').filter(Boolean));
  let all = '';
  for (let attempt = 1; attempt <= 3; attempt++) {
    const r = spawnSync(cmd, args, {
      cwd: WORK, env: cleanEnv(javaHome(launcher) ? { JAVA_HOME: javaHome(launcher) } : {}),
      encoding: 'utf8', timeout: 30 * 60 * 1000, maxBuffer: 64 * 1024 * 1024,
    });
    if (r.error) return { ok: false, errors: 'Gradle could not run: ' + r.error.message, infra: true };
    all = (r.stdout || '') + '\n' + (r.stderr || '');
    if (r.status === 0) return { ok: true };
    const errors = gradleErrors(all);
    if (errors) return { ok: false, errors, infra: false };
    // not a Java error: a download hiccup is worth another try
    if (attempt < 3 && /Too Many Requests|timed out|Connection reset|Could not GET|Read timed out|status code 5\d\d/.test(all)) {
      console.log('  download problem, building again in 20s');
      spawnSync('sleep', ['20']);
      continue;
    }
    break;
  }
  const wrong = all.split('\n');
  const at = wrong.findIndex((l) => l.includes('What went wrong'));
  const detail = (at >= 0 ? wrong.slice(at, at + 12) : wrong.slice(-25)).join('\n').split(WORK + '/').join('').slice(0, 2500);
  return { ok: false, errors: detail, infra: true };
}

function findModJar() {
  const dir = path.join(WORK, 'build', 'libs');
  if (!fs.existsSync(dir)) return '';
  const jars = fs.readdirSync(dir).filter((f) => f.endsWith('.jar') && !/sources|slim|javadoc|dev/.test(f));
  return jars.length ? path.join(dir, jars[0]) : '';
}

function dumpFiles(files) {
  let s = '';
  for (const [p, body] of files) s += '=== FILE: ' + p + ' ===\n' + body;
  return s + '=== END ===';
}

// ---------------------------------------------------------------- main

const KIND_HE = { plugin: 'הפלאגין', forge: 'המוד', fabric: 'המוד' };

async function main() {
  const prompt = (env.PROMPT || '').trim().slice(0, MAX_PROMPT);
  const kind = ['plugin', 'forge', 'fabric'].includes(env.KIND) ? env.KIND : 'plugin';
  const mc = (env.MC_VERSION || '').trim();
  const wantedName = sanitizeName(env.PLUGIN_NAME);
  if (prompt.length < 10) finish({ ok: false, refused: true, error: 'הבקשה קצרה מדי.' });
  if (!MC_RX.test(mc)) finish({ ok: false, error: 'גרסת מיינקראפט לא נתמכת.' });

  const chain = buildChain();
  if (chain.length === 0) {
    finish({ ok: false, error: 'האתר לא הוגדר עד הסוף: חסר מפתח AI (GROQ_API_KEY או GEMINI_API_KEY) ב-GitHub Secrets.' });
  }

  console.log(`kind=${kind} mc=${mc}`);
  const ctx = kind === 'plugin' ? preparePlugin(mc) : kind === 'fabric' ? prepareFabric(mc) : prepareForge(mc);
  console.log('build setup:', JSON.stringify({ ...ctx, example: undefined, mdk: undefined }));

  const base = [
    { role: 'system', content: systemPrompt(kind, mc, wantedName, ctx.example || '') },
    { role: 'user', content: 'Request:\n' + prompt },
  ];
  let messages = base;
  let lastFiles = null;
  let lastErrors = '';
  let model = '';

  for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
    console.log(`--- attempt ${attempt}/${MAX_ATTEMPTS}`);
    const ans = await askAI(chain, messages);
    if (!ans) {
      finish({
        ok: false, attempts: attempt,
        error: 'כל מודלי ה-AI החינמיים עמוסים או שהמכסה היומית שלהם נגמרה. נסו שוב מאוחר יותר.',
      });
    }
    model = ans.model;
    const parsed = parseAnswer(ans.text);
    if (parsed.refuse) finish({ ok: false, refused: true, error: 'ה-AI סירב לבקשה: ' + parsed.refuse });

    const name = wantedName || sanitizeName(parsed.name) || (kind === 'plugin' ? 'AIPlugin' : 'AIMod');
    const modId = name.toLowerCase();
    let problems = checkFiles(parsed.files, kind);
    if (problems.length === 0) {
      if (kind === 'plugin') {
        const ymlProblem = fixPluginYml(parsed.files, name, mc);
        if (ymlProblem) problems.push(ymlProblem);
      } else {
        problems = checkMod(parsed, kind, modId);
      }
    }

    if (problems.length === 0) {
      let built;
      if (kind === 'plugin') { writePlugin(ctx, parsed.files, name, mc); built = compilePlugin(ctx); }
      else if (kind === 'fabric') { writeFabric(ctx, parsed, name, modId, mc); built = compileGradle(ctx, kind); }
      else { writeForge(ctx, parsed, name, modId, mc); built = compileGradle(ctx, kind); }

      if (built.ok) {
        const jar = kind === 'plugin' ? path.join(WORK, 'target', name + '.jar') : findModJar();
        if (!jar || !fs.existsSync(jar)) finish({ ok: false, error: 'הקומפילציה הסתיימה אבל קובץ ה-JAR לא נמצא.' });
        fs.mkdirSync(OUT, { recursive: true });
        const outName = kind === 'plugin' ? name + '.jar' : name + '-' + kind + '-' + mc + '.jar';
        fs.copyFileSync(jar, path.join(OUT, outName));
        const src = kind === 'plugin' ? ['src', 'pom.xml'] : ['src', 'build.gradle', 'settings.gradle', 'gradle.properties'];
        spawnSync('zip', ['-r', '-q', path.join(OUT, name + '-source.zip'), ...src], { cwd: WORK });
        finish({
          ok: true, kind, name, mc, attempts: attempt, model, explain: parsed.explain,
          files: [...parsed.files.keys()],
        });
      }
      if (built.infra) {
        finish({ ok: false, attempts: attempt, error: 'תקלה בכלי הבנייה (לא בקוד). נסו שוב בעוד כמה דקות.', detail: built.errors });
      }
      lastErrors = built.errors;
      lastFiles = parsed.files;
    } else {
      lastErrors = problems.join('\n');
      if (parsed.files.size > 0) lastFiles = parsed.files;
    }
    console.log('problems:\n' + lastErrors);

    messages = base.concat([{
      role: 'user',
      content:
        (lastFiles ? 'Your previous answer was:\n' + dumpFiles(lastFiles).slice(0, 9000) + '\n\n' : '') +
        'It failed with these errors:\n' + lastErrors +
        '\n\nFix every error and answer again in the exact same format: the header lines, ALL files complete, and the EXPLAIN block.',
    }]);
  }

  finish({
    ok: false, attempts: MAX_ATTEMPTS, model,
    error: 'ה-AI לא הצליח לכתוב את ' + KIND_HE[kind] + ' כך שיתקמפל אחרי ' + MAX_ATTEMPTS + ' ניסיונות. נסו לנסח את הבקשה פשוט יותר.',
    detail: lastErrors,
  });
}

main().catch((e) => {
  console.error(e);
  finish({ ok: false, error: 'שגיאה פנימית בבנייה.', detail: String(e && e.message || e).slice(0, 500) });
});
