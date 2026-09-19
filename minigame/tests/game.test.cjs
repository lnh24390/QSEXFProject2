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
  test('shield lasts 15 seconds, refreshes without stacking, and blocks one hit in every mode', () => {
    for (const mode of Object.keys(MODES)) {
      const g = fresh(mode); g.spawnTimer = 100; g.fireTimer = 100;
      g.collect({ kind: 'shield' }); assert.equal(g.shield, 15);
      g.update(.05); assert.ok(Math.abs(g.shield - 14.95) < 1e-9);
      g.pause(); g.update(.05); assert.ok(Math.abs(g.shield - 14.95) < 1e-9); g.resume();
      g.collect({ kind: 'shield' }); assert.equal(g.shield, 15);
      g.invincible = 0; g.damage(); assert.equal(g.shield, 0); assert.equal(g.lives, MODES[mode].lives);
      g.invincible = 0; g.damage(); assert.equal(g.lives, MODES[mode].lives - 1);
      g.reset(mode); g.spawnTimer = 100; g.fireTimer = 100; g.collect({ kind: 'shield' });
      for (let i = 0; i < 299; i++) g.update(.05);
      assert.ok(g.shield > 0); g.update(.05); assert.equal(g.shield, 0);
      g.invincible = 0; g.damage(); assert.equal(g.lives, MODES[mode].lives - 1);
      g.reset(mode); assert.equal(g.shield, 0);
    }
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
  test('kill reward cycle is spread and heal', () => {
    const g = fresh(); for (let i = 0; i < 27; i++) g.reward(enemy(g)); assert.deepEqual(g.pickups.map(p => p.kind), ['spread', 'heal', 'spread']);
  });
  test('regular modes drop one shield every 15 seconds including during boss fights', () => {
    for (const mode of ['chill', 'arcade', 'expert']) {
      const g = fresh(mode); const shields = () => g.pickups.filter(p => p.kind === 'shield').length;
      g.elapsed = 14.98; g.update(.01); assert.equal(shields(), 0);
      g.pause(); g.update(.05); assert.equal(g.nextShield, 15); g.resume();
      g.update(.02); assert.equal(shields(), 1); assert.equal(g.nextShield, 30);
      g.update(.01); assert.equal(shields(), 1);
      g.elapsed = 29.99; g.update(.02); assert.equal(shields(), 2);
      g.elapsed = 44.99; g.update(.02); assert.ok(g.boss); assert.equal(shields(), 3);
      g.boss.hp = 0; g.update(.01); assert.equal(shields(), 3);
      g.reset(mode); assert.equal(g.nextShield, 15); assert.equal(shields(), 0);
    }
  });
  test('impossible drops a shield for each boss defeat and never from time or normal kills', () => {
    const g = fresh('impossible'); const shields = () => g.pickups.filter(p => p.kind === 'shield').length;
    for (let i = 0; i < 27; i++) g.reward(enemy(g)); assert.equal(shields(), 0);
    for (const elapsed of [14.99, 29.99, 44.99]) { g.elapsed = elapsed; g.update(.02); assert.equal(shields(), 0); }
    g.boss.hp = 0; g.update(.01); assert.equal(shields(), 1); assert.equal(g.activeBosses.length, 1);
    g.boss.hp = 0; g.update(.01); assert.equal(shields(), 2); assert.equal(g.boss, null);
    g.update(.01); assert.equal(shields(), 2);
  });
  test('spread pickup adds two angled bullets and expires after ten seconds', () => {
    const g = fresh(); g.collect({ kind: 'spread' }); g.fire(); assert.equal(g.bullets.length, 4); assert.equal(g.bullets.filter(b => b.vx !== 0).length, 2);
    g.spreadTime = .001; g.update(.01); g.bullets = []; g.fire(); assert.equal(g.bullets.length, 2);
  });
  test('healing never exceeds mode maximum; full health grants points', () => {
    const g = fresh('expert'); g.lives = 1; g.collect({ kind: 'heal' }); assert.equal(g.lives, 2); g.collect({ kind: 'heal' }); assert.equal(g.lives, 2); assert.equal(g.score, 250);
  });
  test('boss spawns at 30 seconds and clears preceding hazards', () => {
    const g = fresh(); g.elapsed = 29.99; g.enemies.push(enemy(g)); g.hostile.push({ x: 0, y: 0, vx: 0, vy: 0 }); g.update(.02); assert.ok(g.boss); assert.equal(g.enemies.length, 0); assert.equal(g.hostile.length, 0);
  });
  test('impossible boss waves contain two independent bosses and clear only after both die', () => {
    const g = fresh('impossible'); g.elapsed = 29.99; g.update(.02);
    assert.equal(g.activeBosses.length, 2); assert.equal(g.lives, 1);
    const [left, right] = g.activeBosses;
    assert.ok(left.x < right.x); assert.equal(g.bossWaveHP, left.maxHP + right.maxHP);
    for (const b of g.activeBosses) { b.y = 135; b.shot = 0; }
    g.update(.01); assert.equal(g.hostile.length, 10);
    g.takeEvents(); left.hp = 0; g.update(.01);
    assert.equal(g.boss, right); assert.equal(g.bosses, 1); assert.equal(g.nextBoss, 30);
    assert.equal(g.hostile.length, 10);
    assert.equal(g.takeEvents().find(e => e.type === 'boss-down').remaining, 1);
    right.hp = 0; g.update(.01);
    assert.equal(g.boss, null); assert.equal(g.bosses, 2); assert.equal(g.hostile.length, 0);
    assert.equal(g.nextBoss, g.elapsed + 30);
    g.elapsed = g.nextBoss; g.update(.01); assert.equal(g.activeBosses.length, 2);
    g.reset('impossible'); assert.equal(g.activeBosses.length, 0);
    for (const mode of ['chill', 'arcade', 'expert']) {
      g.reset(mode); g.spawnBoss(); assert.equal(g.activeBosses.length, 1);
    }
  });
  test('all modes drop timed max-health rewards only for wave bosses on every tenth round', () => {
    for (const mode of Object.keys(MODES)) {
      const g = fresh(mode); g.invincible = 100; g.fireTimer = 100;
      const upgrades = () => g.pickups.filter(p => p.kind === 'max-health' && !p.dead);
      for (let round = 1; round <= 30; round++) {
        g.spawnBoss();
        for (const b of g.activeBosses) b.hp = 0;
        g.update(.01);
        const expected = round % 10 === 0 ? 1 : 0;
        assert.equal(upgrades().length, expected);
        if (expected) {
          const item = upgrades()[0], y = item.y;
          g.update(.05); assert.equal(item.y, y); assert.equal(upgrades().length, 1);
          const maximum = g.maxLives;
          g.collect(item); assert.equal(g.maxLives, maximum + 1); assert.equal(g.lives, maximum + 1);
          g.collect(item); assert.equal(g.maxLives, maximum + 1);
          g.invincible = 0; g.damage(); g.collect({ kind: 'heal' }); assert.equal(g.lives, g.maxLives);
          g.invincible = 100;
        }
      }
      assert.equal(g.maxLives, MODES[mode].lives + 3);
      g.reset(mode); assert.equal(g.maxLives, MODES[mode].lives); assert.equal(upgrades().length, 0);
    }
    assert.equal(MODES.impossible.lives, 1);
  });
  test('every tenth round exclusively spawns one wave boss in every difficulty', () => {
    for (const mode of Object.keys(MODES)) {
      const g = fresh(mode); g.invincible = 100; g.fireTimer = 100;
      for (let round = 1; round <= 21; round++) {
        assert.equal(g.round, round); g.elapsed = g.nextBoss; g.update(.01);
        const wave = round % 10 === 0;
        assert.equal(g.activeBosses.length, wave ? 1 : MODES[mode].bossCount || 1);
        assert.ok(g.activeBosses.every(b => b.kind === (wave ? 'wave' : 'sentinel')));
        assert.equal(g.enemies.length, 0);
        if (g.activeBosses.length === 2) { g.boss.hp = 0; g.update(.01); assert.equal(g.round, round); }
        g.boss.hp = 0; g.update(.01); assert.equal(g.round, round + 1);
        assert.equal(g.nextBoss, g.elapsed + 30);
      }
      g.reset(mode); assert.equal(g.round, 1);
    }
  });
  test('max-health reward expires after five gameplay seconds in all modes', () => {
    for (const mode of Object.keys(MODES)) {
      const g = fresh(mode); g.round = 10; g.spawnBoss(); g.boss.hp = 0; g.update(.01);
      const item = g.pickups.find(p => p.kind === 'max-health');
      assert.equal(item.expiresAt, g.elapsed + 5); assert.equal(g.nextBoss, g.elapsed + 30);
      g.pause(); const elapsed = g.elapsed; g.update(.05); assert.equal(g.elapsed, elapsed); g.resume();
      g.elapsed = item.expiresAt - .02; g.update(.01); assert.ok(g.pickups.includes(item));
      g.player.x = item.x; g.player.y = item.y; const max = g.maxLives;
      g.update(.02); assert.ok(!g.pickups.includes(item)); assert.equal(g.maxLives, max);
      g.collect(item); assert.equal(g.maxLives, max);
      g.round = 20; g.spawnBoss(); g.boss.hp = 0; g.player.y = g.height - 44; g.update(.01);
      const next = g.pickups.find(p => p.kind === 'max-health');
      g.elapsed = next.expiresAt - .02; g.player.x = next.x; g.player.y = next.y; g.update(.01);
      assert.equal(g.maxLives, max + 1);
    }
  });
  test('each wave health color equals one normal boss from the preceding round in every mode', () => {
    for (const mode of Object.keys(MODES)) {
      const g = fresh(mode); let previousHP;
      g.invincible = 100; g.fireTimer = 100;
      for (let round = 1; round <= 30; round++) {
        g.spawnBoss();
        if (round % 10 === 0) {
          assert.deepEqual(g.boss.waveHealth, [previousHP, previousHP, previousHP]);
          assert.equal(g.boss.waveMaxHP, previousHP); assert.equal(g.boss.maxHP, previousHP * 3);
        } else previousHP = g.boss.maxHP;
        for (const boss of g.activeBosses) boss.hp = 0;
        g.update(.01);
      }
    }
  });
  test('each living wave transition drops every normal item exactly once in all modes', () => {
    for (const mode of Object.keys(MODES)) {
      const g = fresh(mode); g.round = 10; g.spawnBoss(); const b = g.boss;
      g.fireTimer = 100; g.invincible = 100; b.y = 135;
      for (let wave = 2; wave <= 3; wave++) {
        g.pickups = []; b.hp = b.waveMaxHP * (4 - wave); g.update(.01);
        assert.deepEqual(g.pickups.map(p => p.kind).sort(), ['heal', 'shield', 'spread']);
        g.update(.01); assert.equal(g.pickups.length, 3);
        assert.equal(g.bosses, 0);
      }
      g.pickups = []; b.hp = 0; g.update(.01);
      assert.deepEqual(g.pickups.map(p => p.kind).sort(), mode === 'impossible' ? ['heal', 'max-health', 'shield', 'spread'] : ['heal', 'max-health', 'spread']);
      assert.equal(g.nextBoss, g.elapsed + 30);
    }
  });
  test('wave boss loses only the active color and dies after all three health pools empty', () => {
    const g = fresh(); g.round = 10; g.spawnBoss(); g.fireTimer = 100; g.invincible = 100;
    const b = g.boss; b.y = 135; const hp = b.waveMaxHP;
    assert.deepEqual(b.waveHealth, [hp, hp, hp]);
    const hit = () => { g.bullets.push({x:b.x, y:b.y, vx:0, vy:0}); g.update(.01); };
    hit(); assert.deepEqual(b.waveHealth, [hp - 1, hp, hp]);
    b.hp = hp * 2 + 1; hit();
    assert.deepEqual(b.waveHealth, [0, hp, hp]); assert.equal(b.wave, 2);
    assert.equal(g.bosses, 0); assert.equal(g.round, 10); assert.equal(g.pickups.length, 3);
    b.hp = hp + 1; hit();
    assert.deepEqual(b.waveHealth, [0, 0, hp]); assert.equal(b.wave, 3); assert.equal(g.bosses, 0);
    b.hp = 1; hit(); assert.deepEqual(b.waveHealth, [0, 0, 0]); assert.equal(g.boss, null);
    assert.equal(g.bosses, 1); assert.equal(g.round, 11);
    assert.equal(g.pickups.filter(p => p.kind === 'max-health').length, 1);
  });
  test('wave boss changes from rapid straight fire to normal fan then rapid fan', () => {
    for (const mode of Object.keys(MODES)) {
      const g = fresh(mode); g.round = 10; g.spawnBoss(); g.fireTimer = 100; g.invincible = 100;
      const b = g.boss; b.y = 135; b.shot = 0;
      g.update(.01); assert.equal(b.wave, 1); assert.equal(g.hostile.length, 1);
      assert.equal(g.hostile[0].vx, 0); assert.equal(b.shot, .15);
      g.pause(); g.update(.05); assert.equal(b.shot, .15); g.resume();
      for (let i = 0; i < 4; i++) g.update(.05);
      assert.equal(g.hostile.length, 2);
      g.hostile = []; b.hp = b.maxHP * 2 / 3; b.shot = 0; b.phase = 0;
      g.update(.01); assert.equal(b.wave, 2); assert.equal(g.hostile.length, 5); assert.equal(b.shot, 1.05);
      const directions = g.hostile.map(p => [p.vx, p.vy]);
      g.hostile = []; b.hp = b.maxHP / 3; b.shot = 0; b.phase = 0;
      g.update(.01); assert.equal(b.wave, 3); assert.equal(g.hostile.length, 5); assert.equal(b.shot, .15);
      assert.deepEqual(g.hostile.map(p => [p.vx, p.vy]), directions);
      for (let i = 0; i < 4; i++) g.update(.05);
      assert.equal(g.hostile.length, 10);
      g.reset(mode); g.round = 10; g.spawnBoss(); assert.equal(g.boss.wave, 1);
    }
  });
  test('only stage 100 fires one homing missile every five combat seconds across waves', () => {
    for (const mode of Object.keys(MODES)) {
      const g = fresh(mode); g.round = 100; g.spawnBoss();
      g.fireTimer = 100; g.invincible = 100; g.boss.y = 135;
      const step = count => { for (let i = 0; i < count; i++) g.update(.05); };
      step(99); assert.equal(g.hostile.filter(p => p.homing).length, 0);
      g.pause(); g.update(.05); g.resume();
      step(1); assert.equal(g.hostile.filter(p => p.homing).length, 1);
      const missile = g.hostile.find(p => p.homing);
      const oldAngle = Math.atan2(missile.vy, missile.vx);
      g.player.x = 26; step(1);
      assert.notEqual(Math.atan2(missile.vy, missile.vx), oldAngle);
      g.hostile = []; g.boss.hp = g.boss.waveMaxHP;
      step(98); assert.equal(g.hostile.filter(p => p.homing).length, 0);
      step(1); assert.equal(g.hostile.filter(p => p.homing).length, 1);
    }
    for (const round of [1, 10, 90, 99, 101, 110]) {
      const g = fresh(); g.round = round; g.spawnBoss();
      g.fireTimer = 100; g.invincible = 100; g.boss.y = 135;
      for (let i = 0; i < 120; i++) g.update(.05);
      assert.equal(g.hostile.some(p => p.homing), false);
    }
  });
  test('boss emits fan projectiles and gains aimed shot in second phase', () => {
    const g = fresh(); g.spawnBoss(); g.boss.y = 135; g.boss.shot = 0; g.update(.01); assert.equal(g.hostile.length, 5);
    g.boss.hp = 20; g.boss.shot = 0; g.update(.01); assert.equal(g.hostile.length, 11);
  });
  test('boss clear awards points, clears bullets, drops items and schedules next boss', () => {
    const g = fresh(); g.spawnBoss(); g.boss.hp = 0; g.update(.01); assert.equal(g.boss, null); assert.equal(g.bosses, 1); assert.equal(g.score, 2500); assert.equal(g.hostile.length, 0); assert.equal(g.pickups.length, 2); assert.equal(g.nextBoss, g.elapsed + 30);
  });
  test('shooter enemy aims at player', () => {
    const g = fresh(); g.enemies.push(enemy(g, { kind: 'shooter', shot: 0, x: 200, y: 150, hp: 5 })); g.update(.01); assert.equal(g.hostile.length, 1); assert.ok(g.hostile[0].vx > 0 && g.hostile[0].vy > 0);
  });
  test('restart resets upgrades, hazards, input, timers, scores and boss', () => {
    const g = fresh(); g.collect({ kind: 'spread' }); g.shield = 1; g.score = 9000; g.spawnBoss(); g.dash(); g.move(1, 1); g.reset();
    assert.equal(g.score, 0); assert.equal(g.shield, 0); assert.equal(g.spreadTime, 0); assert.equal(g.boss, null); assert.equal(g.dashCooldown, 0); assert.equal(g.inputX, 0); assert.equal(g.events.length, 0); assert.equal(g.nextBoss, 30);
  });
  test('invalid deltas and stopped states never advance simulation', () => {
    const g = fresh(); for (const dt of [0, -1, NaN, Infinity]) g.update(dt); assert.equal(g.elapsed, 0);
    g.state = 'over'; g.update(.02); assert.equal(g.elapsed, 0); g.state = 'playing'; g.update(100); assert.equal(g.elapsed, .05);
  });
  test('homing missiles survive time, screen bounds and boss changes until contact', () => {
    const g = fresh(); g.fireTimer = 100; g.invincible = 100;
    const missile = { x: -10000, y: -10000, vx: 100, vy: 0, r: 7, homing: true };
    g.hostile.push(missile);
    for (let i = 0; i < 240; i++) g.update(.05);
    assert.ok(g.hostile.includes(missile));
    g.spawnBoss(); assert.ok(g.hostile.includes(missile));
    g.boss.hp = 0; g.update(.01); assert.ok(g.hostile.includes(missile));
    missile.x = g.player.x; missile.y = g.player.y;
    const lives = g.lives; g.update(.01);
    assert.ok(!g.hostile.includes(missile)); assert.equal(g.lives, lives);
    g.invincible = 0;
    const impact = { x: g.player.x, y: g.player.y, vx: 100, vy: 0, r: 7, homing: true };
    g.hostile.push(impact); g.update(.01);
    assert.ok(!g.hostile.includes(impact)); assert.equal(g.lives, lives - 1);
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
