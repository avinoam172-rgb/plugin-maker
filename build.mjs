// Plugin Maker - builder
// Runs inside GitHub Actions: asks a free AI model for plugin code, compiles it
// with Maven, sends compile errors back to the AI for fixing, and writes the
// result into ./out (JAR + source zip + result.json).
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

const ALLOWED_MC = ['1.21.11', '1.21.8', '1.21.4'];
const MAX_ATTEMPTS = 3;
const MAX_PROMPT = 1500;

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
  console.log('RESULT:', JSON.stringify({ ...result, detail: undefined }));
  process.exit(0);
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

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
      if (!e.status && e.name !== 'TimeoutError' && netRetries < 2) {
        // network hiccup (for example a closed idle connection) - try again
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

function systemPrompt(mc, wantedName) {
  return [
    'You write Minecraft server plugins. You receive a request (often in Hebrew) and answer with the complete source of one small plugin.',
    '',
    'Target: Minecraft ' + mc + ', Spigot API (package org.bukkit) so it runs on Spigot, Paper and Purpur. Java 21.',
    '',
    'Rules:',
    '- Use ONLY the Bukkit/Spigot API and the Java standard library. No Paper-only classes, no Adventure/MiniMessage, no NMS, no CraftBukkit classes, no external libraries.',
    '- The main class extends org.bukkit.plugin.java.JavaPlugin.',
    '- Every command is declared in plugin.yml and handled in code. Register every listener.',
    '- Keep it compact: 1 to 4 Java files, at most about 250 lines in total. No placeholder or unfinished code.',
    '- Use only API that really exists in this version. When unsure, use the simplest well-known API.',
    '- Text shown to players is written in the language of the request. Use org.bukkit.ChatColor for colors.',
    '- Never use: network access (java.net), Runtime.exec, ProcessBuilder, reflection, class loading, System.exit, or files outside the plugin data folder.',
    '- Never write hidden behaviour, backdoors, secret operator access, or anything meant to damage a server or steal data.',
    wantedName
      ? '- The plugin name is exactly: ' + wantedName
      : '- Pick a short English plugin name (letters and digits only, starting with a letter).',
    '',
    'If the request is not a request for a Minecraft plugin, or asks for something from the forbidden list, answer with a single line:',
    'REFUSE: <short reason, in the language of the request>',
    '',
    'Otherwise answer in EXACTLY this format and nothing else (no explanations, no markdown code fences):',
    '',
    'PLUGIN_NAME: <name>',
    '=== FILE: src/main/resources/plugin.yml ===',
    '<file content>',
    '=== FILE: src/main/java/<package path>/<ClassName>.java ===',
    '<file content>',
    '=== END ===',
    '',
    'plugin.yml must contain name, version, main, api-version and the commands section when there are commands.',
  ].join('\n');
}

// ---------------------------------------------------------------- parsing and checks

const JAVA_PATH = /^src\/main\/java\/(?:[A-Za-z_][A-Za-z0-9_]*\/)*[A-Za-z_][A-Za-z0-9_]*\.java$/;
const RES_PATH = /^src\/main\/resources\/(?:[A-Za-z0-9_-]+\/)*[A-Za-z0-9_-]+\.(?:yml|yaml|txt|json|properties)$/;

function parseAnswer(text) {
  const t = text.replace(/\r\n/g, '\n').trim();
  const refuse = t.match(/^REFUSE:\s*(.+)$/m);
  if (refuse && !/=== FILE:/.test(t)) return { refuse: refuse[1].trim().slice(0, 300) };

  const nameMatch = t.match(/^PLUGIN_NAME:\s*([A-Za-z0-9_]+)/m);
  const files = new Map();
  const re = /^=== FILE:\s*(.+?)\s*===[ \t]*$/gm;
  const marks = [];
  let m;
  while ((m = re.exec(t)) !== null) marks.push({ p: m[1], start: m.index, bodyStart: re.lastIndex });
  for (let i = 0; i < marks.length; i++) {
    let body = t.slice(marks[i].bodyStart, i + 1 < marks.length ? marks[i + 1].start : t.length);
    const end = body.search(/^=== END ===/m);
    if (end >= 0) body = body.slice(0, end);
    body = body.replace(/^\s*\n/, '').replace(/\s+$/, '');
    body = body.replace(/^```[A-Za-z]*\n/, '').replace(/\n```$/, '');
    const p = marks[i].p.replace(/\\/g, '/').replace(/^\.?\//, '');
    files.set(p, body + '\n');
  }
  return { name: nameMatch ? nameMatch[1] : '', files };
}

// returns a list of problems (empty = fine)
function checkFiles(files) {
  const problems = [];
  if (files.size === 0) return ['The answer contained no "=== FILE: ... ===" blocks. Use the exact output format.'];
  if (files.size > 12) problems.push('Too many files (max 12).');
  let total = 0;
  let javaCount = 0;
  for (const [p, body] of files) {
    total += body.length;
    if (p.includes('..') || /META-INF/i.test(p) || !(JAVA_PATH.test(p) || RES_PATH.test(p))) {
      problems.push('File path not allowed: ' + p.slice(0, 120) +
        ' (only src/main/java/**.java and src/main/resources/*.yml|txt|json|properties)');
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
  if (!files.has('src/main/resources/plugin.yml')) problems.push('Missing src/main/resources/plugin.yml.');
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

// ---------------------------------------------------------------- build

function cleanEnv() {
  const out = {};
  for (const [k, v] of Object.entries(env)) {
    if (/KEY|TOKEN|SECRET|PASSWORD|PROMPT/i.test(k)) continue;
    out[k] = v;
  }
  return out;
}

function writeProject(files, name, mc) {
  fs.rmSync(WORK, { recursive: true, force: true });
  fs.mkdirSync(WORK, { recursive: true });
  let pom = fs.readFileSync(path.join(HERE, 'pom.xml'), 'utf8');
  pom = pom.replace('<plugin.name>Plugin</plugin.name>', '<plugin.name>' + name + '</plugin.name>')
           .replace('<mc.version>1.21.11</mc.version>', '<mc.version>' + mc + '</mc.version>');
  fs.writeFileSync(path.join(WORK, 'pom.xml'), pom);
  for (const [p, body] of files) {
    const full = path.join(WORK, p);
    if (!full.startsWith(WORK + path.sep)) throw new Error('bad path');
    fs.mkdirSync(path.dirname(full), { recursive: true });
    fs.writeFileSync(full, body);
  }
}

function compile() {
  const r = spawnSync('mvn', ['-B', '-q', '-ntp', 'package', '-DskipTests'], {
    cwd: WORK, env: cleanEnv(), encoding: 'utf8', timeout: 6 * 60 * 1000, maxBuffer: 32 * 1024 * 1024,
  });
  if (r.error) return { ok: false, errors: 'Maven could not run: ' + r.error.message, infra: true };
  if (r.status === 0) return { ok: true };
  const all = (r.stdout || '') + '\n' + (r.stderr || '');
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
  const errors = lines.join('\n').slice(0, 3000) || all.slice(-1500);
  const infra = !/\.java/.test(errors) && /resolve|transfer|Could not find artifact|Connection|timed out/i.test(errors);
  return { ok: false, errors, infra };
}

function dumpFiles(files) {
  let s = '';
  for (const [p, body] of files) s += '=== FILE: ' + p + ' ===\n' + body;
  return s + '=== END ===';
}

// ---------------------------------------------------------------- main

async function main() {
  const prompt = (env.PROMPT || '').trim().slice(0, MAX_PROMPT);
  const mc = ALLOWED_MC.includes(env.MC_VERSION) ? env.MC_VERSION : ALLOWED_MC[0];
  const wantedName = sanitizeName(env.PLUGIN_NAME);
  if (prompt.length < 10) finish({ ok: false, error: 'הבקשה קצרה מדי.' });

  const chain = buildChain();
  if (chain.length === 0) {
    finish({ ok: false, error: 'האתר לא הוגדר עד הסוף: חסר מפתח AI (GROQ_API_KEY או GEMINI_API_KEY) ב-GitHub Secrets.' });
  }

  const base = [
    { role: 'system', content: systemPrompt(mc, wantedName) },
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

    let problems = checkFiles(parsed.files);
    let name = wantedName || sanitizeName(parsed.name) || 'AIPlugin';
    if (problems.length === 0) {
      const ymlProblem = fixPluginYml(parsed.files, name, mc);
      if (ymlProblem) problems.push(ymlProblem);
    }

    if (problems.length === 0) {
      writeProject(parsed.files, name, mc);
      const built = compile();
      if (built.ok) {
        const jar = path.join(WORK, 'target', name + '.jar');
        if (!fs.existsSync(jar)) finish({ ok: false, error: 'הקומפילציה הסתיימה אבל קובץ ה-JAR לא נמצא.' });
        fs.mkdirSync(OUT, { recursive: true });
        fs.copyFileSync(jar, path.join(OUT, name + '.jar'));
        spawnSync('zip', ['-r', '-q', path.join(OUT, name + '-source.zip'), 'src', 'pom.xml'], { cwd: WORK });
        finish({ ok: true, name, mc, attempts: attempt, model, files: [...parsed.files.keys()] });
      }
      if (built.infra) {
        finish({ ok: false, attempts: attempt, error: 'תקלה זמנית בהורדת ספריות הקומפילציה. נסו שוב בעוד כמה דקות.', detail: built.errors });
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
        '\n\nFix every error and answer again with ALL files, complete, in the exact same format.',
    }]);
  }

  finish({
    ok: false, attempts: MAX_ATTEMPTS, model,
    error: 'ה-AI לא הצליח לכתוב קוד שמתקמפל אחרי ' + MAX_ATTEMPTS + ' ניסיונות. נסו לנסח את הבקשה פשוט יותר.',
    detail: lastErrors,
  });
}

main().catch((e) => {
  console.error(e);
  finish({ ok: false, error: 'שגיאה פנימית בבנייה.', detail: String(e && e.message || e).slice(0, 500) });
});
