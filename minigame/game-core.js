/* Pure simulation: no DOM, audio, timers or storage. Browser + Node compatible. */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.NeonCore = factory();
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';
  const clamp = (n, lo, hi) => Math.max(lo, Math.min(hi, n));
  const PLAYER_MARGIN = 26; // Half of the rendered fighter sprite width.
  const MODES = Object.freeze({
    chill: { label: '편안하게', lives: 5, speed: .72, spawn: 1.25, bossHP: .75 },
    arcade: { label: '아케이드', lives: 3, speed: 1, spawn: 1, bossHP: 1 },
    expert: { label: '하드코어', lives: 2, speed: 1.25, spawn: .8, bossHP: 1.3 },
    impossible: { label: '불가능', lives: 1, speed: 1.25, spawn: .8, bossHP: 1.3, bossCount: 2 }
  });
  // Swept collision prevents fast projectiles passing through a target between frames.
  function segmentHit(ax, ay, bx, by, cx, cy, radius) {
    const dx = bx - ax, dy = by - ay, len = dx * dx + dy * dy;
    const t = len ? clamp(((cx - ax) * dx + (cy - ay) * dy) / len, 0, 1) : 0;
    return Math.hypot(cx - ax - t * dx, cy - ay - t * dy) <= radius;
  }
  class Game {
    constructor({ width = 1000, height = 600, random = Math.random } = {}) {
      this.random = random; this.width = width; this.height = height;
      this.state = 'menu'; this.mode = 'arcade'; this.reset(); this.state = 'menu';
    }
    rand(a, b) { return a + this.random() * (b - a); }
    reset(mode = this.mode) {
      this.mode = MODES[mode] ? mode : 'arcade'; this.config = MODES[this.mode];
      Object.assign(this, { state: 'playing', elapsed: 0, score: 0, lives: this.config.lives, maxLives: this.config.lives,
        kills: 0, crystals: 0, bosses: 0, round: 1, combo: 0, maxCombo: 0, comboTime: 0,
        sector: 1, spawnTimer: .8, fireTimer: .1, dashCooldown: 0, dashTime: 0,
        invincible: 1.5, shield: 0, spreadTime: 0, nextBoss: 30, nextHeal: 20, nextShield: 15, activeBosses: [], bossWaveHP: 0,
        drops: 0, target: null, inputX: 0, inputY: 0 });
      this.player = { x: this.width / 2, y: this.height * .77, vx: 0, vy: -1 };
      this.enemies = []; this.bullets = []; this.hostile = []; this.gems = [];
      this.pickups = []; this.events = [];
    }
    emit(type, data = {}) { this.events.push({ type, ...data }); }
    takeEvents() { return this.events.splice(0); }
    get boss() { return this.activeBosses[0] || null; }
    get multiplier() { return Math.min(5, 1 + Math.floor(this.combo / 5)); }
    pause() { if (this.state === 'playing') { this.state = 'paused'; this.clearInput(); } }
    resume() { if (this.state === 'paused') this.state = 'playing'; }
    clearInput() { this.inputX = this.inputY = 0; this.target = null; }
    move(x, y) { this.inputX = x; this.inputY = y; this.target = null; }
    aim(x, y) { this.target = { x: clamp(x, PLAYER_MARGIN, this.width - PLAYER_MARGIN), y: clamp(y, 90, this.height - 44) }; }
    resize(width, height) {
      if (!(width > 60 && height > 150)) return;
      const sx = width / this.width, sy = height / this.height;
      for (const p of [this.player, ...this.enemies, ...this.bullets, ...this.hostile, ...this.gems, ...this.pickups, ...this.activeBosses]) {
        p.x *= sx; p.y *= sy;
      }
      this.width = width; this.height = height; this.clearInput();
      this.player.x = clamp(this.player.x, PLAYER_MARGIN, width - PLAYER_MARGIN);
      this.player.y = clamp(this.player.y, 90, height - 44);
    }
    dash() {
      if (this.state !== 'playing' || this.dashCooldown > 0) return false;
      let dx = this.inputX, dy = this.inputY;
      if (this.target) { dx = this.target.x - this.player.x; dy = this.target.y - this.player.y; }
      const len = Math.hypot(dx, dy);
      this.dashX = len > .1 ? dx / len : this.player.vx;
      this.dashY = len > .1 ? dy / len : this.player.vy;
      this.dashTime = .2; this.dashCooldown = 3; this.invincible = Math.max(this.invincible, .55);
      this.emit('dash', { x: this.player.x, y: this.player.y }); return true;
    }
    damage() {
      if (this.state !== 'playing' || this.invincible > 0) return false;
      this.invincible = 1.6;
      if (this.shield) { this.shield = 0; this.emit('shield-break', { ...this.player }); return true; }
      this.lives--; this.combo = 0; this.comboTime = 0;
      this.emit('damage', { x: this.player.x, y: this.player.y });
      if (this.lives <= 0) { this.state = 'over'; this.clearInput(); this.emit('over'); }
      return true;
    }
    spawnEnemy() {
      const roll = this.random();
      const kind = this.elapsed > 18 && roll < .2 ? 'shooter' : this.elapsed > 9 && roll < .43 ? 'heavy' : 'scout';
      this.enemies.push({ x: this.rand(25, this.width - 25), y: -30,
        r: kind === 'heavy' ? 21 : kind === 'shooter' ? 16 : 13,
        hp: kind === 'heavy' ? 4 : kind === 'shooter' ? 2 : 1,
        speed: (this.rand(55, 90) + Math.min(this.elapsed * .8, 100)) * this.config.speed,
        phase: this.rand(0, Math.PI * 2), rotation: 0, shot: this.rand(1.3, 2.4), kind, dead: false });
    }
    fire() {
      for (const offset of [-6, 6]) this.bullets.push({ x: this.player.x + offset, y: this.player.y - 16, vx: 0, vy: -600 });
      if (this.spreadTime > 0) for (const sign of [-1, 1]) this.bullets.push({ x: this.player.x, y: this.player.y - 12, vx: sign * 175, vy: -560 });
      this.emit('fire');
    }
    spawnBoss() {
      const kind = this.round % 10 === 0 ? 'wave' : 'sentinel';
      // The preceding normal round started before its one or two bosses were defeated.
      const healthLevel = kind === 'wave' ? Math.max(0, this.bosses - (this.config.bossCount || 1)) : this.bosses;
      const hp = Math.round((100 + healthLevel * 35) * this.config.bossHP);
      const count = kind === 'wave' ? 1 : this.config.bossCount || 1;
      this.activeBosses = Array.from({ length: count }, (_, index) => ({
        x: this.width * (index + .5) / count, y: -65, r: 42, hp, maxHP: hp,
        kind, wave: 1, rewardedWaves: 0, shot: 1.4, homingTimer: this.round === 100 ? 5 : null, phase: 0, lane: index, laneCount: count
      }));
      if (kind === 'wave') {
        const boss = this.boss;
        boss.waveMaxHP = hp;
        boss.waveHealth = [hp, hp, hp];
        boss.maxHP = hp * 3;
        Object.defineProperty(boss, 'hp', {
          enumerable: true,
          get() { return this.waveHealth.reduce((sum, health) => sum + health, 0); },
          set(value) {
            const total = clamp(value, 0, this.maxHP);
            this.waveHealth = [0, 1, 2].map(index => clamp(total - (2 - index) * this.waveMaxHP, 0, this.waveMaxHP));
          }
        });
      }
      this.bossWaveHP = this.activeBosses.reduce((sum, boss) => sum + boss.maxHP, 0);
      this.enemies = []; this.hostile = this.hostile.filter(b => b.homing && !b.dead);
      this.emit('boss', { count, kind, round: this.round });
    }
    updateBossWave(boss) {
      if (boss.kind !== 'wave' || boss.hp <= 0) return;
      const wave = boss.waveHealth.findIndex(health => health > 0) + 1;
      while (boss.rewardedWaves < wave - 1) {
        boss.rewardedWaves++;
        ['spread', 'shield', 'heal'].forEach((kind, index) => {
          this.pickups.push({ x: clamp(boss.x + (index - 1) * 35, 25, this.width - 25), y: boss.y, kind });
        });
      }
      if (wave !== boss.wave) {
        boss.wave = wave;
        boss.shot = Math.min(boss.shot, wave === 2 ? 1.05 : .15);
        this.emit('boss-wave', { wave });
      }
    }
    aimedShot(x, y, speed) {
      const a = Math.atan2(this.player.y - y, this.player.x - x);
      this.hostile.push({ x, y, vx: Math.cos(a) * speed, vy: Math.sin(a) * speed, r: 5 });
    }
    reward(enemy) {
      enemy.dead = true; this.kills++; this.combo++; this.comboTime = 3.5;
      this.maxCombo = Math.max(this.maxCombo, this.combo);
      const points = (enemy.kind === 'heavy' ? 150 : enemy.kind === 'shooter' ? 100 : 50) * this.multiplier;
      this.score += points;
      this.emit('kill', { x: enemy.x, y: enemy.y, kind: enemy.kind, points });
      this.gems.push({ x: enemy.x, y: enemy.y, phase: this.rand(0, 6) });
      if (this.kills % 9 === 0) {
        const kinds = ['spread', 'heal'];
        this.pickups.push({ x: enemy.x, y: enemy.y, kind: kinds[this.drops++ % kinds.length] });
      }
    }
    collect(pickup) {
      if (pickup.dead) return;
      if (pickup.expiresAt !== undefined && this.elapsed >= pickup.expiresAt) { pickup.dead = true; return; }
      let healed = 0, bonus = 0;
      if (pickup.kind === 'spread') this.spreadTime = 10;
      if (pickup.kind === 'shield') this.shield = 15;
      if (pickup.kind === 'max-health') { this.maxLives++; this.lives = Math.min(this.maxLives, this.lives + 1); }
      if (pickup.kind === 'heal') {
        if (this.lives < this.maxLives) { this.lives++; healed = 1; }
        else { this.score += 250; bonus = 250; }
      }
      pickup.dead = true; this.emit('pickup', { x: pickup.x, y: pickup.y, kind: pickup.kind, healed, bonus });
    }
    update(dt) {
      if (this.state !== 'playing' || !Number.isFinite(dt) || dt <= 0) return;
      dt = Math.min(dt, .05); // Browser uses a fixed 1/120s step; guard callers after long stalls.
      this.elapsed += dt;
      if (this.elapsed >= this.nextHeal) {
        this.nextHeal = this.elapsed + 20;
        if (!this.pickups.some(p => p.kind === 'heal' && !p.dead)) {
          this.pickups.push({ x: this.rand(40, this.width - 40), y: 105, kind: 'heal' });
          this.emit('heal-drop');
        }
      }
      if (this.mode !== 'impossible' && this.elapsed >= this.nextShield) {
        this.nextShield += 15;
        this.pickups.push({ x: this.rand(40, this.width - 40), y: 105, kind: 'shield' });
      }
      for (const timer of ['dashCooldown', 'dashTime', 'invincible', 'shield', 'spreadTime', 'comboTime']) this[timer] = Math.max(0, this[timer] - dt);
      if (!this.comboTime) this.combo = 0;
      const next = 1 + Math.floor(this.elapsed / 20);
      if (next !== this.sector) { this.sector = next; this.emit('sector', { sector: next }); }
      let dx = this.inputX, dy = this.inputY, speed = Math.min(360, this.width * .95);
      if (this.target) { dx = this.target.x - this.player.x; dy = this.target.y - this.player.y; speed = Math.min(speed, Math.hypot(dx, dy) * 10); }
      const len = Math.hypot(dx, dy);
      if (len > .1) { dx /= len; dy /= len; this.player.vx = dx; this.player.vy = dy; } else dx = dy = 0;
      if (this.dashTime > 0) { dx = this.dashX; dy = this.dashY; speed = Math.min(1050, this.width * 2.5); }
      this.player.x = clamp(this.player.x + dx * speed * dt, PLAYER_MARGIN, this.width - PLAYER_MARGIN);
      this.player.y = clamp(this.player.y + dy * speed * dt, 90, this.height - 44);
      this.fireTimer -= dt;
      if (this.fireTimer <= 0) { this.fireTimer += this.spreadTime > 0 ? .11 : .15; this.fire(); }
      if (!this.boss && this.elapsed >= this.nextBoss) this.spawnBoss();
      this.spawnTimer -= dt;
      if (!this.boss && this.spawnTimer <= 0) {
        this.spawnTimer = Math.max(.25, .85 - this.elapsed * .004) * this.config.spawn;
        this.spawnEnemy();
      }
      for (const b of this.bullets) { b.px = b.x; b.py = b.y; b.x += b.vx * dt; b.y += b.vy * dt; }
      for (const e of this.enemies) {
        e.y += e.speed * dt; e.x = clamp(e.x + Math.sin(this.elapsed * 2 + e.phase) * 27 * dt, e.r, this.width - e.r);
        e.rotation += dt * (e.kind === 'heavy' ? .6 : -1);
        if (e.kind === 'shooter' && e.y > 85 && e.y < this.height - 90) {
          e.shot -= dt;
          if (e.shot <= 0) { e.shot = 2; this.aimedShot(e.x, e.y, 145 * this.config.speed); }
        }
        for (const b of this.bullets) if (!b.dead && !e.dead && segmentHit(b.px, b.py, b.x, b.y, e.x, e.y, e.r + 3)) {
          b.dead = true; e.hp--;
          if (e.hp <= 0) this.reward(e);
          else this.emit('hit', { x: b.x, y: b.y });
        }
        if (!e.dead && Math.hypot(this.player.x - e.x, this.player.y - e.y) < e.r + 9 && this.damage()) e.dead = true;
        if (this.state === 'over') return;
      }
      for (const b of [...this.activeBosses]) {
        b.phase += dt;
        b.y = Math.min(135, b.y + dt * 70); b.x = this.width * (b.lane + .5) / b.laneCount + Math.sin(b.phase * .65) * (b.laneCount === 1 ? Math.max(20, this.width * .3) : Math.max(0, this.width / b.laneCount / 2 - 65));
        if (b.y >= 110) {
          this.updateBossWave(b);
          if (b.homingTimer !== null) {
            b.homingTimer -= dt;
            if (b.homingTimer <= 1e-9) {
              b.homingTimer += 5;
              this.aimedShot(b.x, b.y + 25, 155 * this.config.speed);
              Object.assign(this.hostile[this.hostile.length - 1], { homing: true, r: 7 });
            }
          }
          b.shot -= dt;
          if (b.shot <= 0) {
            const waveBoss = b.kind === 'wave';
            b.shot = waveBoss ? (b.wave === 2 ? 1.05 : .15) : b.hp < b.maxHP * .45 ? .65 : 1.05;
            if (waveBoss && b.wave === 1) {
              this.hostile.push({ x: b.x, y: b.y + 25, vx: 0, vy: 155 * this.config.speed, r: 5 });
            } else {
              for (let i = -2; i <= 2; i++) {
                const angle = Math.PI / 2 + i * .28 + Math.sin(b.phase) * .18;
                this.hostile.push({ x: b.x, y: b.y + 25, vx: Math.cos(angle) * 155 * this.config.speed, vy: Math.sin(angle) * 155 * this.config.speed, r: 5 });
              }
            }
            if (!waveBoss && b.hp < b.maxHP * .45) this.aimedShot(b.x, b.y, 190 * this.config.speed);
          }
        }
        for (const shot of this.bullets) if (!shot.dead && segmentHit(shot.px, shot.py, shot.x, shot.y, b.x, b.y, b.r + 3)) { shot.dead = true; b.hp--; this.emit('hit', { x: shot.x, y: shot.y }); }
        this.updateBossWave(b);
        if (b.hp <= 0) {
          this.score += 2500 * this.multiplier; this.bosses++;
          this.activeBosses = this.activeBosses.filter(active => active !== b);
          this.emit('boss-down', { x: b.x, y: b.y, remaining: this.activeBosses.length });
          if (this.mode === 'impossible') this.pickups.push({ x: b.x - 30, y: b.y, kind: 'shield' });
          this.pickups.push({ x: b.x, y: b.y, kind: 'heal' }, { x: b.x + 30, y: b.y, kind: 'spread' });
          if (!this.boss) {
            this.hostile = this.hostile.filter(shot => shot.homing && !shot.dead); this.nextBoss = this.elapsed + 30;
            if (b.kind === 'wave') {
              this.pickups.push({ x: this.width / 2, y: Math.min(200, this.height / 2), kind: 'max-health', expiresAt: this.elapsed + 5 });
              this.emit('max-health-drop');
            }
            this.round++;
          }
        } else if (Math.hypot(this.player.x - b.x, this.player.y - b.y) < b.r + 9) this.damage();
        if (this.state === 'over') return;
      }
      for (const b of this.hostile) {
        if (b.homing) {
          const speed = Math.hypot(b.vx, b.vy);
          const angle = Math.atan2(b.vy, b.vx);
          const target = Math.atan2(this.player.y - b.y, this.player.x - b.x);
          const turn = Math.atan2(Math.sin(target - angle), Math.cos(target - angle));
          const heading = angle + clamp(turn, -1.8 * dt, 1.8 * dt);
          b.vx = Math.cos(heading) * speed; b.vy = Math.sin(heading) * speed;
        }
        const ox = b.x, oy = b.y; b.x += b.vx * dt; b.y += b.vy * dt;
        if (segmentHit(ox, oy, b.x, b.y, this.player.x, this.player.y, b.r + 8)) {
          const damaged = this.damage();
          if (damaged || b.homing) b.dead = true;
        }
        if (this.state === 'over') return;
      }
      for (const p of [...this.gems, ...this.pickups]) {
        if (p.expiresAt !== undefined && this.elapsed >= p.expiresAt) { p.dead = true; continue; }
        p.y += (p.kind === 'max-health' ? 0 : p.kind ? 42 : 55) * dt;
        const distance = Math.hypot(p.x - this.player.x, p.y - this.player.y);
        if (distance < (p.kind ? 95 : 135)) { p.x += (this.player.x - p.x) * dt * 6; p.y += (this.player.y - p.y) * dt * 6; }
        if (Math.hypot(p.x - this.player.x, p.y - this.player.y) < 24) {
          if (p.kind) this.collect(p);
          else { p.dead = true; this.score += 100 * this.multiplier; this.crystals++; this.emit('gem', { x: p.x, y: p.y }); }
        }
      }
      this.bullets = this.bullets.filter(b => !b.dead && b.y > -30 && b.x > -20 && b.x < this.width + 20);
      this.enemies = this.enemies.filter(e => !e.dead && e.y < this.height + 40);
      this.hostile = this.hostile.filter(b => !b.dead && (b.homing || (b.y > -40 && b.y < this.height + 30 && b.x > -30 && b.x < this.width + 30)));
      this.gems = this.gems.filter(g => !g.dead && g.y < this.height + 20);
      this.pickups = this.pickups.filter(p => !p.dead && p.y < this.height + 20);
    }
  }
  return { Game, MODES, segmentHit, clamp };
});
