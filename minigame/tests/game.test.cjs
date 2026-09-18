'use strict';
const assert = require('node:assert/strict');
const { Game, MODES, segmentHit } = require('../game-core.js');

function runTests() {
  const results = [];
  const test = (name, action) => { action(); results.push(name); };
  const fresh = (mode = 'arcade', options = {}) => { const g = new Game({ random: () => .5, ...options }); g.reset(mode); return g; };
  const tick = (g, seconds) => { for (let i = 0; i < Math.round(seconds * 120); i++) { g.update(1 / 120); g.takeEvents(); } };
  const enemy = (g, overrides = {}) => ({ x: g.player.x, y: g.player.y - 80, hp: 1, r: 13, kind: 'scout', speed: 0, phase: 0, rotation: 0, shot: 2, ...overrides });

  test('all difficulties initialize with correct lives; unknown modes fall back', () => {
    for (const mode of Object.keys(MODES)) { const g = fresh(mode); assert.equal(g.lives, MODES[mode].lives); assert.equal(g.state, 'playing'); }
    assert.equal(fresh('invalid').mode, 'arcade');
  });
  test('paused simulation freezes every gameplay timer and projectile', () => {
    const g = fresh(); g.dash(); g.fire(); g.takeEvents(); g.pause(); const before = JSON.stringify(g); tick(g, 2); assert.equal(JSON.stringify(g), before);
  });
  test('resume clears no progress and permits simulation', () => {
    const g = fresh(); g.score = 700; g.pause(); g.resume(); g.update(.01); assert.equal(g.score, 700); assert.equal(g.elapsed, .01);
  });
  test('diagonal movement has the same speed as horizontal movement', () => {
    const a = fresh(), b = fresh(); a.move(1, 0); b.move(1, 1);
    const ax = a.player.x, ay = a.player.y; a.update(.05); b.update(.05);
    assert.ok(Math.abs(Math.hypot(a.player.x - ax, a.player.y - ay) - Math.hypot(b.player.x - ax, b.player.y - ay)) < 1e-6);
  });
  test('movement and resizing respect arena boundaries', () => {
    const g = fresh(); g.move(-1, -1); tick(g, 3); assert.ok(g.player.x >= 18 && g.player.y >= 90);
    g.resize(320, 490); assert.ok(g.player.x <= 302 && g.player.y <= 446);
    g.resize(0, 0); assert.equal(g.width, 320);
  });
  test('pointer target converges without oscillating', () => {
    const g = fresh(); g.aim(560, 350); tick(g, 1); assert.ok(Math.hypot(g.player.x - 560, g.player.y - 350) < 1);
  });
  test('dash follows new keyboard input and cannot be retriggered during cooldown', () => {
    const g = fresh(); g.move(1, 0); assert.equal(g.dash(), true); const x = g.player.x; g.update(.02); assert.ok(g.player.x > x + 15); assert.equal(g.dash(), false);
    g.enemies = []; tick(g, 3.1); assert.equal(g.dash(), true);
  });
  test('dash protects against damage and works while stationary at pointer target', () => {
    const g = fresh(); g.invincible = 0; g.aim(g.player.x, g.player.y); g.dash(); const y = g.player.y; g.update(.02); assert.ok(g.player.y < y); assert.equal(g.damage(), false); assert.equal(g.lives, 3);
  });
  test('normal damage grants grace time; three distinct hits end a run', () => {
    const g = fresh(); g.invincible = 0; assert.equal(g.damage(), true); assert.equal(g.damage(), false); assert.equal(g.lives, 2);
    g.invincible = 0; g.damage(); g.invincible = 0; g.damage(); assert.equal(g.state, 'over'); assert.equal(g.lives, 0); assert.equal(g.damage(), false);
  });
  test('shield absorbs one hit without losing a life or combo', () => {
    const g = fresh(); g.shield = 1; g.combo = 7; g.invincible = 0; g.damage(); assert.equal(g.shield, 0); assert.equal(g.lives, 3); assert.equal(g.combo, 7);
  });
  test('enemy collision uses player hitbox; far enemies do not damage', () => {
    const g = fresh(); g.invincible = 0; g.fireTimer = 10;
    g.enemies.push(enemy(g, { x: g.player.x + 100, y: g.player.y, hp: 100 })); g.update(.01); assert.equal(g.lives, 3);
    g.enemies.push(enemy(g, { y: g.player.y, hp: 100 })); g.update(.01); assert.equal(g.lives, 2);
  });
  test('fast projectiles use swept collision, including hostile bullets', () => {
    assert.equal(segmentHit(0, 0, 0, 100, 1, 50, 5), true); assert.equal(segmentHit(0, 0, 0, 100, 10, 50, 5), false);
    const g = fresh(); g.fireTimer = 10; g.enemies.push(enemy(g, { x: 200, y: 220 }));
    g.bullets.push({ x: 200, y: 260, vx: 0, vy: -2000 }); g.update(.04); assert.equal(g.kills, 1);
    g.invincible = 0; g.hostile.push({ x: g.player.x, y: g.player.y - 40, vx: 0, vy: 2000, r: 5 }); g.update(.04); assert.equal(g.lives, 2);
  });
  test('one enemy rewards once even when hit by multiple projectiles', () => {
    const g = fresh(); const e = enemy(g); g.enemies.push(e); g.fireTimer = 10;
    for (let i = 0; i < 3; i++) g.bullets.push({ x: e.x, y: e.y + 5, vx: 0, vy: -600 });
    g.update(.01); assert.equal(g.kills, 1); assert.equal(g.score, 50); assert.equal(g.gems.length, 1);
  });
  test('combos increase multiplier, cap at five, and expire', () => {
    const g = fresh(); for (let i = 0; i < 25; i++) g.reward(enemy(g)); assert.equal(g.multiplier, 5); assert.equal(g.maxCombo, 25);
    g.comboTime = .001; g.update(.01); assert.equal(g.combo, 0); assert.equal(g.multiplier, 1);
  });
  test('crystals attract, collect once, and apply score multiplier', () => {
    const g = fresh(); g.combo = 5; g.comboTime = 3; g.gems.push({ x: g.player.x, y: g.player.y, phase: 0 });
    g.update(.01); assert.equal(g.score, 200); assert.equal(g.crystals, 1); assert.equal(g.gems.length, 0); g.update(.01); assert.equal(g.score, 200);
  });
  test('guaranteed powerup cycle is spread, shield, heal', () => {
    const g = fresh(); for (let i = 0; i < 27; i++) g.reward(enemy(g)); assert.deepEqual(g.pickups.map(p => p.kind), ['spread', 'shield', 'heal']);
  });
  test('spread pickup adds two angled bullets and expires after ten seconds', () => {
    const g = fresh(); g.collect({ kind: 'spread' }); g.fire(); assert.equal(g.bullets.length, 4); assert.equal(g.bullets.filter(b => b.vx !== 0).length, 2);
    g.spreadTime = .001; g.update(.01); g.bullets = []; g.fire(); assert.equal(g.bullets.length, 2);
  });
  test('healing never exceeds mode maximum; full health grants points', () => {
    const g = fresh('expert'); g.lives = 1; g.collect({ kind: 'heal' }); assert.equal(g.lives, 2); g.collect({ kind: 'heal' }); assert.equal(g.lives, 2); assert.equal(g.score, 250);
  });
  test('boss spawns at 45 seconds and clears preceding hazards', () => {
    const g = fresh(); g.elapsed = 44.99; g.enemies.push(enemy(g)); g.hostile.push({ x: 0, y: 0, vx: 0, vy: 0 }); g.update(.02); assert.ok(g.boss); assert.equal(g.enemies.length, 0); assert.equal(g.hostile.length, 0);
  });
  test('boss emits fan projectiles and gains aimed shot in second phase', () => {
    const g = fresh(); g.spawnBoss(); g.boss.y = 135; g.boss.shot = 0; g.update(.01); assert.equal(g.hostile.length, 5);
    g.boss.hp = 20; g.boss.shot = 0; g.update(.01); assert.equal(g.hostile.length, 11);
  });
  test('boss clear awards points, clears bullets, drops items and schedules next boss', () => {
    const g = fresh(); g.spawnBoss(); g.boss.hp = 0; g.update(.01); assert.equal(g.boss, null); assert.equal(g.bosses, 1); assert.equal(g.score, 2500); assert.equal(g.hostile.length, 0); assert.equal(g.pickups.length, 2); assert.equal(g.nextBoss, g.elapsed + 45);
  });
  test('shooter enemy aims at player', () => {
    const g = fresh(); g.enemies.push(enemy(g, { kind: 'shooter', shot: 0, x: 200, y: 150, hp: 5 })); g.update(.01); assert.equal(g.hostile.length, 1); assert.ok(g.hostile[0].vx > 0 && g.hostile[0].vy > 0);
  });
  test('restart resets upgrades, hazards, input, timers, scores and boss', () => {
    const g = fresh(); g.collect({ kind: 'spread' }); g.shield = 1; g.score = 9000; g.spawnBoss(); g.dash(); g.move(1, 1); g.reset();
    assert.equal(g.score, 0); assert.equal(g.shield, 0); assert.equal(g.spreadTime, 0); assert.equal(g.boss, null); assert.equal(g.dashCooldown, 0); assert.equal(g.inputX, 0); assert.equal(g.events.length, 0); assert.equal(g.nextBoss, 45);
  });
  test('invalid deltas and stopped states never advance simulation', () => {
    const g = fresh(); for (const dt of [0, -1, NaN, Infinity]) g.update(dt); assert.equal(g.elapsed, 0);
    g.state = 'over'; g.update(.02); assert.equal(g.elapsed, 0); g.state = 'playing'; g.update(100); assert.equal(g.elapsed, .05);
  });
  test('offscreen entities are removed', () => {
    const g = fresh(); g.fireTimer = 10; g.bullets = [{ x: 10, y: -100, vx: 0, vy: -600 }]; g.hostile = [{ x: -90, y: 100, vx: -100, vy: 0, r: 5 }]; g.gems = [{ x: 20, y: 800 }]; g.pickups = [{ x: 20, y: 800, kind: 'shield' }]; g.update(.01);
    for (const list of ['bullets', 'hostile', 'gems', 'pickups']) assert.equal(g[list].length, 0);
  });
  test('five-minute simulation stays finite and entity counts remain bounded', () => {
    let seed = 1234; const g = fresh('expert', { random: () => { seed = (1664525 * seed + 1013904223) >>> 0; return seed / 4294967296; } });
    for (let i = 0; i < 36000; i++) {
      g.invincible = 1; if (i % 120 === 0) g.aim(g.boss ? g.boss.x : g.width / 2, g.height * .78);
      g.update(1 / 120); g.takeEvents();
      assert.ok(Number.isFinite(g.player.x) && Number.isFinite(g.player.y));
      assert.ok(g.bullets.length < 200 && g.hostile.length < 200 && g.enemies.length < 100);
    }
    assert.ok(g.elapsed > 299); assert.ok(g.bosses > 0); assert.ok(g.kills > 0);
  });
  return results;
}
module.exports = runTests;
if (require.main === module) {
  const results = runTests();
  results.forEach(name => console.log(`PASS ${name}`));
  console.log(`\n${results.length} tests passed.`);
}
