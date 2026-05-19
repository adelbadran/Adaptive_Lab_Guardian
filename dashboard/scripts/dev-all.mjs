import { spawn } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const root = path.resolve(__dirname, '..');

const processes = [
  {
    name: 'api',
    command: process.execPath,
    args: ['server.mjs'],
  },
  {
    name: 'vite',
    command: process.execPath,
    args: [path.join('node_modules', 'vite', 'bin', 'vite.js'), '--port=3000', '--host=0.0.0.0'],
  },
  {
    name: 'ai-bridge',
    command: 'python',
    args: ['-u', path.resolve(root, '..', 'ai', 'mqtt_client.py')],
  },
];

const children = processes.map(({ name, command, args }) => {
  const child = spawn(command, args, {
    cwd: root,
    env: process.env,
    shell: false,
    stdio: ['inherit', 'pipe', 'pipe'],
  });

  child.stdout.on('data', (chunk) => {
    process.stdout.write(`[${name}] ${chunk}`);
  });
  child.stderr.on('data', (chunk) => {
    process.stderr.write(`[${name}] ${chunk}`);
  });
  child.on('exit', (code) => {
    if (code !== 0 && code !== null) {
      process.stderr.write(`[${name}] exited with code ${code}\n`);
    }
  });

  return child;
});

function shutdown() {
  for (const child of children) {
    if (!child.killed) child.kill();
  }
}

process.on('SIGINT', () => {
  shutdown();
  process.exit(0);
});
process.on('SIGTERM', () => {
  shutdown();
  process.exit(0);
});
