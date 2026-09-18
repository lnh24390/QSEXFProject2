'use strict';
// DOM contract tests. Canvas calls are checked for finite coordinates, not pixels.
// These complement, and do not replace, manual browser layout checks.
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const core = require('../game-core.js');

function createPage({ width = 1000, height = 620, touch = false, storageBlocked = false } = {}) {
  const elements = new Map(), windowEvents = new Map(), documentEvents = new Map(), storage = new Map();
  let game, frame, now = 0, pointerCaptures = 0;
  const ctx = new Proxy({}, { get(target, key) {
    if (key in target) return target[key];
    if (key.startsWith('create')) return () => ({ addColorStop() {} });
    return (...args) => { args.filter(v => typeof v === 'number').forEach(v => assert.ok(Number.isFinite(v), `nonfinite canvas ${key}`)); };
  }, set(target, key, value) { target[key] = value; return true; } });
  class Element {
    constructor(id, tag = 'DIV') { this.id = id; this.tagName = tag; this.style = {}; this.dataset = {}; this.attrs = {}; this.listeners = {}; this.classes = new Set(); this.textContent = ''; this.hidden = false; this.disabled = false; this._html = ''; this.classList = { add: name => this.classes.add(name), remove: name => this.classes.delete(name), toggle: (name, force) => { const yes = force ?? !this.classes.has(name); yes ? this.classes.add(name) : this.classes.delete(name); return yes; }, contains: name => this.classes.has(name) }; }
    addEventListener(name, fn) { (this.listeners[name] ||= []).push(fn); }
    setAttribute(name, value) { this.attrs[name] = String(value); }
    focus() { document.activeElement = this; }
    matches(selector) { return selector.split(',').some(tag => tag.toUpperCase() === this.tagName); }
    closest(selector) { return selector === 'button' && this.tagName === 'BUTTON' ? this : null; }
    getBoundingClientRect() { return { width, height, left: 0, top: 0 }; }
    getContext() { return ctx; }
    setPointerCapture() { pointerCaptures++; }
    set innerHTML(value) { this._html = value; parse(value); }
    get innerHTML() { return this._html; }
    emit(name, data = {}) { const event = { target: this, preventDefault() {}, stopPropagation() {}, ...data }; for (const fn of this.listeners[name] || []) fn(event); }
  }
  function parse(text) {
    for (const match of text.matchAll(/<([\w-]+)\b[^>]*\bid="([^"]+)"[^>]*>/g)) elements.set(match[2], new Element(match[2], match[1].toUpperCase()));
  }
  const html = fs.readFileSync(path.join(__dirname, '../index.html'), 'utf8'); parse(html);
  const panel = elements.get('panel'); panel._html = html.match(/<div class="splash-content" id="panel">([\s\S]*?)<\/div><\/div>\s*<div class="arena-footer">/)[1];
  const modes = Object.keys(core.MODES).map(mode => { const button = new Element('', 'BUTTON'); button.dataset.mode = mode; return button; });
  const document = { body: new Element('body', 'BODY'), activeElement: null, hidden: false, getElementById: id => { assert.ok(elements.has(id), `HTML missing #${id}`); return elements.get(id); }, querySelectorAll: () => modes, addEventListener: (name, fn) => documentEvents.set(name, fn) };
  const sandbox = { console, Math, Set, String, Number, Array, document, devicePixelRatio: 2,
    window: { addEventListener: (name, fn) => windowEvents.set(name, fn) }, performance: { now: () => now },
    localStorage: { getItem(key) { if (storageBlocked) throw Error('blocked'); return storage.get(key) ?? null; }, setItem(key, value) { if (storageBlocked) throw Error('blocked'); storage.set(key, value); } },
    matchMedia: query => ({ matches: query.includes('coarse') && touch }), ResizeObserver: class { observe() {} },
    requestAnimationFrame: callback => { frame = callback; },
    NeonCore: { ...core, Game: class extends core.Game { constructor() { super({ random: () => .5 }); game = this; } } }
  };
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../game.js'), 'utf8'), sandbox, { filename: 'game.js' });
  const get = id => elements.get(id);
  const key = (key, options = {}) => windowEvents.get('keydown')({ key, target: document.activeElement || document.body, preventDefault() {}, repeat: false, ...options });
  const frames = count => { for (let i = 0; i < count; i++) { now += 1000 / 60; frame(now); } };
  return { game, get, key, frames, modes, document, storage, windowEvents, documentEvents, captured: () => pointerCaptures };
}
function runTests() {
  const results = [], test = (name, action) => { action(); results.push(name); };
  test('menu binds existing HTML; start focuses canvas and runs render loop', () => {
    const p = createPage(); p.get('start').emit('click'); p.frames(120); assert.equal(p.game.state, 'playing'); assert.equal(p.document.activeElement.id, 'game'); assert.ok(p.game.elapsed > 1.9); assert.equal(p.get('run-status').hidden, false);
  });
  test('keyboard move, dash, pause and resume reach the real controller', () => {
    const p = createPage(); p.key('Enter'); const x = p.game.player.x; p.key('d'); p.frames(10); assert.ok(p.game.player.x > x); p.key(' '); p.frames(1); assert.ok(p.game.dashCooldown > 0);
    p.key('p'); const elapsed = p.game.elapsed; p.frames(60); assert.equal(p.game.elapsed, elapsed); assert.equal(p.game.state, 'paused'); p.get('resume').emit('click'); assert.equal(p.game.state, 'playing');
  });
  test('leaving tab pauses and releases keyboard input', () => {
    const p = createPage(); p.key('Enter'); p.key('d'); p.windowEvents.get('blur')(); assert.equal(p.game.state, 'paused'); assert.equal(p.game.inputX, 0);
  });
  test('sound and effects controls exist independently of overlays', () => {
    const p = createPage(); p.get('effects').emit('click'); assert.equal(p.get('effects').attrs['aria-pressed'], 'true'); assert.equal(p.storage.get('neon-drift-effects'), 'low'); assert.ok(p.get('sound').listeners.click.length);
  });
  test('difficulty selection, stored record and initial lives agree', () => {
    const p = createPage(); p.modes[0].emit('click'); p.get('start').emit('click'); assert.equal(p.game.mode, 'chill'); assert.equal(p.game.lives, 5); assert.equal(p.storage.get('neon-drift-mode'), 'chill');
  });
  test('game over updates record and retry creates a clean run', () => {
    const p = createPage(); p.key('Enter'); p.game.score = 1234; p.game.lives = 1; p.game.invincible = 0; p.game.fireTimer = 10;
    p.game.enemies.push({ x: p.game.player.x, y: p.game.player.y, r: 15, hp: 100, speed: 0, phase: 0, rotation: 0, kind: 'heavy' });
    p.frames(2); assert.equal(p.game.state, 'over'); assert.ok(p.get('panel').innerHTML.includes('001234')); assert.equal(p.storage.get('neon-drift-best'), '1234');
    p.get('restart').emit('click'); assert.equal(p.game.score, 0); assert.equal(p.game.lives, 3); assert.equal(p.game.boss, null);
  });
  test('main menu can select a different mode and start again', () => {
    const p = createPage(); p.key('Enter'); p.key('p'); p.get('home').emit('click'); assert.equal(p.game.state, 'menu'); p.modes[2].emit('click'); p.get('start').emit('click'); assert.equal(p.game.mode, 'expert');
  });
  test('touch uses relative dragging and ignores a second pointer', () => {
    const p = createPage({ width: 360, height: 540, touch: true }); p.get('start').emit('click'); const canvas = p.get('game'); const initial = p.game.player.x;
    canvas.emit('pointerdown', { pointerId: 1, pointerType: 'touch', clientX: 50, clientY: 350 }); assert.equal(p.game.target, null);
    canvas.emit('pointerdown', { pointerId: 2, pointerType: 'touch', clientX: 300, clientY: 350 }); assert.equal(p.captured(), 1);
    canvas.emit('pointermove', { pointerId: 1, pointerType: 'touch', clientX: 80, clientY: 350 }); p.frames(15); assert.ok(p.game.player.x > initial);
    canvas.emit('pointercancel', { pointerId: 1 }); assert.equal(p.game.target, null);
    p.get('mobile-dash').emit('pointerdown'); assert.ok(p.game.dashCooldown > 0);
  });
  test('storage failure does not prevent starting or finishing', () => {
    const p = createPage({ storageBlocked: true }); p.key('Enter'); p.frames(5); p.get('effects').emit('click'); assert.equal(p.game.state, 'playing');
  });
  test('desktop and mobile render with finite canvas arguments through boss phase', () => {
    for (const width of [320, 390, 768, 1280]) { const p = createPage({ width, height: 620 }); p.key('Enter'); p.game.elapsed = 44.99; p.game.invincible = 100; p.frames(240); assert.ok(p.game.boss); assert.equal(p.get('boss-hud').hidden, false); assert.ok(p.get('boss-fill').style.width.endsWith('%')); }
  });
  return results;
}
module.exports = runTests;
if (require.main === module) { const results = runTests(); results.forEach(name => console.log(`PASS ${name}`)); console.log(`\n${results.length} UI contract tests passed.`); }
