(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const { Game, MODES, clamp } = NeonCore;
  const canvas = $('game'), ctx = canvas.getContext('2d'), arena = $('arena');
  const playerSprite = $('player-sprite');
  const game = new Game();
  const touch = matchMedia('(pointer: coarse)').matches;
  const saved = (key, fallback) => { try { return localStorage.getItem(key) ?? fallback; } catch { return fallback; } };
  const save = (key, value) => { try { localStorage.setItem(key, value); } catch {} };
  let mode = saved('neon-drift-mode', 'arcade'); if (!MODES[mode]) mode = 'arcade';
  let reduced = saved('neon-drift-effects', matchMedia('(prefers-reduced-motion: reduce)').matches ? 'low' : 'full') === 'low';
  let sound = false, audio, master;
  let width = 1000, height = 600, clock = 0, last = 0, accumulator = 0, hudClock = 0;
  let particles = [], rings = [], labels = [], stars = [], shake = 0, flash = 0, toastTime = 0;
  let pointer = null, dragPoint = null;
  const keys = new Set();
  const menuHTML = $('panel').innerHTML;
  const pad = n => Math.floor(n).toString().padStart(6, '0');
  const timeLabel = t => `${String(Math.floor(t / 60)).padStart(2, '0')}:${String(Math.floor(t % 60)).padStart(2, '0')}`;
  const rand = (a, b) => a + Math.random() * (b - a);
  const bestKey = () => mode === 'arcade' ? 'neon-drift-best' : `neon-drift-best-${mode}`;
  const bestScore = () => { const n = Number(saved(bestKey(), '0')); return Number.isFinite(n) && n > 0 ? n : 0; };
  function resize() {
    const rect = canvas.getBoundingClientRect(); width = rect.width; height = rect.height;
    const dpr = Math.min(devicePixelRatio || 1, 2);
    canvas.width = Math.round(width * dpr); canvas.height = Math.round(height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0); game.resize(width, height);
    stars = Array.from({ length: Math.round(width * height / 5000) }, () => ({ x: rand(0, width), y: rand(0, height), r: rand(.5, 1.6), s: rand(8, 30), a: rand(.2, .7) }));
  }
  new ResizeObserver(resize).observe(arena); resize();
  function tone(freq = 440, duration = .1, type = 'sine', volume = .03) {
    if (!sound || !audio || audio.state !== 'running') return;
    const oscillator = audio.createOscillator(), gain = audio.createGain();
    oscillator.type = type;
    oscillator.frequency.setValueAtTime(freq, audio.currentTime);
    oscillator.frequency.exponentialRampToValueAtTime(Math.max(30, freq * .5), audio.currentTime + duration);
    gain.gain.setValueAtTime(volume, audio.currentTime);
    gain.gain.exponentialRampToValueAtTime(.001, audio.currentTime + duration);
    oscillator.connect(gain); gain.connect(master);
    oscillator.onended = () => { oscillator.disconnect(); gain.disconnect(); };
    oscillator.start(); oscillator.stop(audio.currentTime + duration);
  }
  $('sound').addEventListener('click', async () => {
    try {
      if (!audio) { audio = new (window.AudioContext || window.webkitAudioContext)(); master = audio.createGain(); master.gain.value = .65; master.connect(audio.destination); }
      await audio.resume(); sound = !sound;
    } catch { sound = false; announce('이 브라우저에서는 소리를 재생할 수 없어요'); }
    $('sound').setAttribute('aria-pressed', String(sound));
    $('sound').setAttribute('aria-label', sound ? '소리 끄기' : '소리 켜기');
    $('sound').title = sound ? '소리 끄기' : '소리 켜기';
    $('sound-path').setAttribute('d', sound ? 'M15 8c3 2 3 6 0 8m3-11c5 4 5 10 0 14' : 'm16 9 6 6m0-6-6 6');
    tone(660, .13);
  });
  function effectsUI() {
    document.body.classList.toggle('reduced-effects', reduced);
    $('effects').setAttribute('aria-pressed', String(reduced));
    $('effects').setAttribute('aria-label', reduced ? '화면 효과 늘리기' : '화면 효과 줄이기');
    $('effects').title = reduced ? '화면 효과 늘리기' : '화면 효과 줄이기';
  }
  $('effects').addEventListener('click', () => { reduced = !reduced; save('neon-drift-effects', reduced ? 'low' : 'full'); effectsUI(); }); effectsUI();
  function burst(x, y, color, count = 20, power = 140) {
    for (let i = 0; i < (reduced ? count / 4 : count); i++) {
      const angle = rand(0, Math.PI * 2), speed = rand(25, power), life = rand(.25, .7);
      particles.push({ x, y, vx: Math.cos(angle) * speed, vy: Math.sin(angle) * speed, life, max: life, size: rand(1, 3), color });
    }
    rings.push({ x, y, r: 4, life: .45, color });
    if (particles.length > 500) particles.splice(0, particles.length - 500);
    if (rings.length > 40) rings.shift();
  }
  function announce(message) { $('toast').textContent = message; $('toast').classList.add('show'); toastTime = 2.5; }
  function clearInput() { keys.clear(); game.clearInput(); pointer = null; dragPoint = null; }
  function chooseMode(next) {
    mode = next; save('neon-drift-mode', mode);
    document.querySelectorAll('[data-mode]').forEach(button => { const chosen = button.dataset.mode === mode; button.classList.toggle('selected', chosen); button.setAttribute('aria-pressed', String(chosen)); });
    $('best').textContent = pad(bestScore());
    updateHealth(MODES[mode].lives, MODES[mode].lives);
  }
  function bindMenu() {
    document.querySelectorAll('[data-mode]').forEach(button => button.addEventListener('click', () => chooseMode(button.dataset.mode)));
    $('start').addEventListener('click', start); chooseMode(mode);
  }
  function start() {
    if (sound && audio) audio.resume().catch(() => {});
    game.reset(mode); clearInput(); particles = []; rings = []; labels = []; shake = flash = 0;
    accumulator = 0; last = performance.now();
    $('overlay').classList.add('hidden'); $('pause').style.display = 'grid'; $('pause').textContent = 'Ⅱ'; $('pause').setAttribute('aria-label', '일시정지');
    $('run-status').hidden = false; $('mobile-dash').style.display = touch ? 'block' : 'none';
    canvas.focus({ preventScroll: true });
    announce('빛을 모으세요 · 45초 뒤 첫 보스'); updateHUD(); tone(660, .2);
  }
  function home() {
    game.reset(mode); game.state = 'menu'; clearInput(); particles = []; rings = []; labels = [];
    $('panel').innerHTML = menuHTML; $('overlay').classList.remove('hidden');
    $('pause').style.display = 'none'; $('mobile-dash').style.display = 'none'; $('run-status').hidden = true; $('boss-hud').hidden = true;
    $('toast').classList.remove('show'); toastTime = 0; $('score').textContent = '000000';
    $('sector').textContent = 'SECTOR 01 / THE BEGINNING'; bindMenu(); $('start').focus({ preventScroll: true });
  }
  function togglePause() {
    if (game.state === 'playing') {
      game.pause(); clearInput(); $('mobile-dash').style.display = 'none';
      $('overlay').classList.remove('hidden');
      $('panel').innerHTML = '<div class="badge"><i></i> TAKE A BREATHER</div><h2 class="results-title">잠시, 숨 고르기.</h2><p class="pause-note">방향키 / WASD / 마우스로 이동<br>Space로 대시 · 연속 처치로 최대 5배 점수<br>강화 아이템: 확산탄 10초 / 보호막 / 회복</p><button class="start-button" id="resume">계속 플레이 <span>↗</span></button><button class="secondary-button" id="home">메인 화면으로</button><div class="enter-hint">P 또는 Esc로 계속</div>';
      $('resume').addEventListener('click', togglePause); $('home').addEventListener('click', home);
      $('pause').textContent = '▶'; $('pause').setAttribute('aria-label', '계속 플레이'); $('resume').focus({ preventScroll: true });
    } else if (game.state === 'paused') {
      if (sound && audio) audio.resume().catch(() => {});
      game.resume(); clearInput(); last = performance.now(); accumulator = 0;
      $('overlay').classList.add('hidden'); $('pause').textContent = 'Ⅱ'; $('pause').setAttribute('aria-label', '일시정지');
      $('mobile-dash').style.display = touch ? 'block' : 'none'; canvas.focus({ preventScroll: true });
    }
  }
  $('pause').addEventListener('click', togglePause);
  // Called by the offline Android shell; no native object is exposed to JavaScript.
  window.neonAndroidPause = () => {
    clearInput();
    if (game.state === 'playing') togglePause();
    if (audio && audio.state === 'running') audio.suspend().catch(() => {});
  };
  window.neonAndroidBack = () => {
    if (game.state === 'playing') { togglePause(); return true; }
    if (game.state === 'paused' || game.state === 'over') { home(); return true; }
    return false;
  };
  function gameOver() {
    const record = game.score > bestScore(); if (record) save(bestKey(), String(game.score));
    $('best').textContent = pad(bestScore()); $('pause').style.display = 'none'; $('mobile-dash').style.display = 'none';
    $('boss-hud').hidden = true; $('toast').classList.remove('show'); toastTime = 0; clearInput();
    $('overlay').classList.remove('hidden');
    $('panel').innerHTML = `<div class="badge"><i></i> ${MODES[mode].label} / RUN COMPLETE</div><h2 class="results-title">멋진 드리프트였어요.</h2><div class="results-score">${pad(game.score)}</div><div class="new-record">${record ? '✦ 새로운 최고 기록!' : '한 번 더, 조금 더 멀리.'}</div><div class="result-grid"><div><strong>${timeLabel(game.elapsed)}</strong><small>생존 시간</small></div><div><strong>${game.kills}</strong><small>적 격파</small></div><div><strong>${game.maxCombo}</strong><small>최대 콤보</small></div></div><p class="result-summary">보스 ${game.bosses}회 격파 · 크리스털 ${game.crystals}개</p><button class="start-button" id="restart">다시 도전 <span>↗</span></button><button class="secondary-button" id="home">난이도 변경</button>`;
    $('restart').addEventListener('click', start); $('home').addEventListener('click', home); $('restart').focus({ preventScroll: true });
    updateHUD(); tone(100, .5, 'sawtooth', .04);
  }
  const moveKeys = ['arrowup', 'arrowdown', 'arrowleft', 'arrowright', 'w', 'a', 's', 'd'];
  function keyboardMove() { game.move((keys.has('d') || keys.has('arrowright') ? 1 : 0) - (keys.has('a') || keys.has('arrowleft') ? 1 : 0), (keys.has('s') || keys.has('arrowdown') ? 1 : 0) - (keys.has('w') || keys.has('arrowup') ? 1 : 0)); }
  window.addEventListener('keydown', event => {
    if (event.ctrlKey || event.metaKey || event.altKey || event.target.matches('input,textarea,select')) return;
    const k = event.key.toLowerCase();
    if ((game.state === 'playing' && (moveKeys.includes(k) || k === ' ')) || k === 'escape') event.preventDefault();
    if ((k === 'p' || k === 'escape') && !event.repeat) { togglePause(); return; }
    if (k === 'enter' && !event.repeat && !event.target.closest('button')) {
      if (game.state === 'menu' || game.state === 'over') start(); else if (game.state === 'paused') togglePause();
    }
    if (game.state !== 'playing') return;
    if (k === ' ' && !event.repeat) game.dash();
    if (moveKeys.includes(k)) { keys.add(k); keyboardMove(); }
  });
  window.addEventListener('keyup', event => { const k = event.key.toLowerCase(); if (keys.delete(k)) keyboardMove(); });
  window.addEventListener('blur', () => { clearInput(); if (game.state === 'playing') togglePause(); });
  document.addEventListener('visibilitychange', () => { if (document.hidden) { clearInput(); if (game.state === 'playing') togglePause(); } });
  function pointerPoint(event) { const r = canvas.getBoundingClientRect(); return { x: event.clientX - r.left, y: event.clientY - r.top }; }
  canvas.addEventListener('pointerdown', event => {
    if (game.state !== 'playing' || pointer !== null) return;
    pointer = event.pointerId; dragPoint = pointerPoint(event); canvas.setPointerCapture(pointer);
    if (event.pointerType === 'mouse') game.aim(dragPoint.x, dragPoint.y);
  });
  canvas.addEventListener('pointermove', event => {
    if (game.state !== 'playing') return;
    const point = pointerPoint(event);
    if (event.pointerType === 'mouse') { keys.clear(); game.aim(point.x, point.y); }
    else if (event.pointerId === pointer && dragPoint) {
      const anchor = game.target || game.player;
      game.aim(anchor.x + point.x - dragPoint.x, anchor.y + point.y - dragPoint.y); dragPoint = point;
    }
  });
  function release(event) { if (pointer === event.pointerId) { pointer = null; dragPoint = null; game.target = null; } }
  canvas.addEventListener('pointerup', release); canvas.addEventListener('pointercancel', release); canvas.addEventListener('lostpointercapture', release);
  canvas.addEventListener('pointerleave', () => { if (pointer === null) game.target = null; });
  $('mobile-dash').addEventListener('pointerdown', event => { event.preventDefault(); event.stopPropagation(); game.dash(); });
  function updateHealth(current, maximum) {
    const health = clamp(current, 0, maximum), ratio = health / maximum;
    $('health-value').textContent = `${health} / ${maximum}`;
    $('health-fill').style.width = `${ratio * 100}%`;
    $('health').classList.toggle('low', ratio <= 1 / 3);
    $('health').classList.toggle('warning', ratio > 1 / 3 && ratio <= .5);
    $('health-bar').setAttribute('aria-valuemax', String(maximum));
    $('health-bar').setAttribute('aria-valuenow', String(health));
    $('health-bar').setAttribute('aria-valuetext', `체력 ${health} / ${maximum}`);
  }
  function updateHUD() {
    $('score').textContent = pad(game.score);
    updateHealth(game.lives, game.config.lives);
    $('run-clock').textContent = timeLabel(game.elapsed);
    $('combo').textContent = game.combo ? `×${game.multiplier} / ${game.combo} COMBO` : '×1';
    $('combo').classList.toggle('hot', game.multiplier > 1);
    $('power-status').textContent = [game.shield ? '◈ 보호막' : '', game.spreadTime > 0 ? `확산탄 ${Math.ceil(game.spreadTime)}s` : ''].filter(Boolean).join(' · ') || '기본 사격';
    $('sector').textContent = `SECTOR ${String(game.sector).padStart(2, '0')} / ${game.boss ? 'BOSS FIGHT' : `BOSS IN ${Math.max(0, Math.ceil(game.nextBoss - game.elapsed))}s`}`;
    $('dash-fill').style.width = `${(1 - game.dashCooldown / 3) * 100}%`;
    $('dash-label').textContent = game.dashCooldown > 0 ? `DASH ${game.dashCooldown.toFixed(1)}s` : 'DASH READY';
    $('mobile-dash').textContent = game.dashCooldown > 0 ? `${game.dashCooldown.toFixed(1)}s` : 'DASH ↗';
    $('mobile-dash').disabled = game.dashCooldown > 0;
    $('boss-hud').hidden = !game.boss || !['playing', 'paused'].includes(game.state);
    if (game.boss) { const percent = clamp(game.boss.hp / game.boss.maxHP * 100, 0, 100); $('boss-fill').style.width = `${percent}%`; $('boss-percent').textContent = `${Math.ceil(percent)}%`; }
  }
  function handleEvents() {
    for (const event of game.takeEvents()) {
      const { type, x, y } = event;
      if (type === 'fire') tone(900, .035, 'sine', .006);
      if (type === 'kill') { burst(x, y, event.kind === 'heavy' ? '#ffb26f' : '#b9a0ff', 22); labels.push({ x, y, text: `+${event.points}`, life: .8, color: '#d5c5ff' }); tone(310, .09, 'triangle'); }
      if (type === 'hit') burst(x, y, '#ffbd8a', 3, 40);
      if (type === 'gem') { burst(x, y, '#b5ffda', 7, 65); tone(1200, .08, 'sine', .017); }
      if (type === 'dash') { burst(x, y, '#b5ffda', 26, 220); tone(520, .25, 'triangle'); }
      if (type === 'damage') { burst(x, y, '#ff754a', 38, 230); shake = reduced ? 0 : 9; flash = reduced ? 0 : .18; tone(100, .25, 'sawtooth'); }
      if (type === 'shield-break') { burst(x, y, '#81dfff', 32, 190); announce('보호막이 충격을 막았어요'); tone(420, .2, 'triangle'); }
      if (type === 'pickup') {
        burst(x, y, '#b5ffda', event.kind === 'heal' ? 38 : 25, 150);
        const healMessage = event.healed ? '체력 +1 회복!' : '체력 가득 · 보너스 +250점';
        announce({ spread: '확산탄 가동 · 10초', shield: '보호막 장착 · 피격 1회 방어', heal: healMessage }[event.kind]);
        if (event.kind === 'heal') {
          labels.push({ x: game.player.x, y: game.player.y - 26, text: event.healed ? 'HP +1' : '+250', life: 1.2, color: '#b5ffda' });
          updateHUD();
        }
        tone(1300, .25);
      }
      if (type === 'heal-drop') announce('회복 캡슐 등장 · 초록색 +를 모으세요');
      if (type === 'sector') announce(`SECTOR ${String(event.sector).padStart(2, '0')} — KEEP GOING`);
      if (type === 'boss') { announce('WARNING · 센티널 접근 중'); tone(150, .7, 'triangle'); }
      if (type === 'boss-down') { burst(x, y, '#ffba85', 80, 310); burst(x, y, '#b5ffda', 50, 230); shake = reduced ? 0 : 7; announce('BOSS CLEAR · 다음 전투까지 45초'); tone(880, .5, 'triangle'); }
      if (type === 'over') gameOver();
    }
  }
  function polygon(x, y, r, sides, rotation, color, fill) {
    ctx.save(); ctx.translate(x, y); ctx.rotate(rotation); ctx.beginPath();
    for (let i = 0; i < sides; i++) { const a = i * Math.PI * 2 / sides - Math.PI / 2; i ? ctx.lineTo(Math.cos(a) * r, Math.sin(a) * r) : ctx.moveTo(Math.cos(a) * r, Math.sin(a) * r); }
    ctx.closePath(); ctx.fillStyle = fill; ctx.fill(); ctx.strokeStyle = color; ctx.lineWidth = 1.5; ctx.shadowBlur = reduced ? 0 : 15; ctx.shadowColor = color; ctx.stroke(); ctx.restore();
  }
  function background() {
    ctx.fillStyle = '#101020'; ctx.fillRect(0, 0, width, height);
    const glow = ctx.createRadialGradient(width * .54, height * .4, 0, width * .54, height * .4, width * .65);
    glow.addColorStop(0, game.boss ? '#68283865' : '#45236665'); glow.addColorStop(.5, '#25204a45'); glow.addColorStop(1, '#0c142100'); ctx.fillStyle = glow; ctx.fillRect(0, 0, width, height);
    const mint = ctx.createRadialGradient(width * .14, height * .8, 0, width * .14, height * .8, width * .4); mint.addColorStop(0, '#134b4e30'); mint.addColorStop(1, '#134b4e00'); ctx.fillStyle = mint; ctx.fillRect(0, 0, width, height);
    ctx.save(); ctx.strokeStyle = '#a783dd16'; ctx.lineWidth = 1; const horizon = height * .53;
    for (let i = -12; i < 15; i++) { ctx.beginPath(); ctx.moveTo(width / 2 + i * 30, horizon); ctx.lineTo(width / 2 + i * 160, height); ctx.stroke(); }
    for (let i = 0; i < 12; i++) { const t = (i / 12 + (reduced ? 0 : clock * .035)) % 1; const y = horizon + t * t * (height - horizon); ctx.globalAlpha = t * .8; ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y); ctx.stroke(); } ctx.restore();
    for (const star of stars) { ctx.globalAlpha = star.a; ctx.fillStyle = '#d7c9ff'; ctx.fillRect(star.x, (star.y + (reduced ? 0 : clock * star.s * .25)) % height, star.r, star.r); } ctx.globalAlpha = 1;
    if (game.state === 'menu') {
      const r = Math.min(width * .29, height * .42); ctx.save(); ctx.translate(width * .5, height * .43); ctx.rotate(-.4); ctx.strokeStyle = '#bc91ef20';
      for (const factor of [1.7, 1.95]) { ctx.beginPath(); ctx.ellipse(0, 0, r * factor, r * .48 * factor, 0, 0, Math.PI * 2); ctx.stroke(); } ctx.restore();
      polygon(width * .16, height * .36, 22, 4, reduced ? 0 : clock * .15, '#b79eff80', '#b79eff07');
      polygon(width * .83, height * .66, 30, 3, reduced ? 0 : -clock * .12, '#ff986e80', '#ff986e08');
      polygon(width * .8, height * .26, 8, 4, 0, '#b5ffda', '#b5ffda20'); polygon(width * .2, height * .78, 6, 4, 0, '#b5ffda88', '#b5ffda10');
    }
  }
  function drawPlayer() {
    const p = game.player, dashing = game.dashTime > 0;
    ctx.save(); ctx.translate(p.x, p.y); ctx.rotate(p.vx * .12);
    if (game.invincible > 0 && !dashing) ctx.globalAlpha = reduced ? .65 : .45 + .55 * Math.abs(Math.sin(clock * 12));
    const color = dashing ? '#b5ffda' : '#e6d9ff'; ctx.shadowBlur = reduced ? 0 : 22; ctx.shadowColor = color;
    const spriteReady = playerSprite.complete && playerSprite.naturalWidth > 0;
    const tailY = spriteReady ? 20 : 8;
    const flame = ctx.createLinearGradient(0, tailY, 0, tailY + 35); flame.addColorStop(0, '#bca2ff'); flame.addColorStop(1, '#af85ff00');
    ctx.fillStyle = flame; ctx.beginPath(); ctx.moveTo(-5, tailY); ctx.lineTo(0, tailY + 25 + (reduced ? 0 : Math.sin(clock * 40) * 7)); ctx.lineTo(5, tailY); ctx.fill();
    if (spriteReady) {
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = 'high';
      ctx.drawImage(playerSprite, -26, -26, 52, 52);
    } else {
      ctx.beginPath(); ctx.moveTo(0, -19); ctx.lineTo(13, 13); ctx.lineTo(0, 7); ctx.lineTo(-13, 13); ctx.closePath(); ctx.fillStyle = '#30264a'; ctx.fill(); ctx.strokeStyle = color; ctx.lineWidth = 1.8; ctx.stroke();
      ctx.fillStyle = '#f4ebff'; ctx.beginPath(); ctx.moveTo(0, -8); ctx.lineTo(4, 5); ctx.lineTo(-4, 5); ctx.closePath(); ctx.fill();
    }
    if (dashing || game.shield) { ctx.globalAlpha = .8; ctx.strokeStyle = game.shield ? '#81dfff' : '#b5ffda'; ctx.beginPath(); ctx.arc(0, 0, spriteReady ? 31 : 27, 0, Math.PI * 2); ctx.stroke(); } ctx.restore();
  }
  function render(dt) {
    const paused = game.state === 'paused'; const visualDt = paused ? 0 : dt;
    ctx.save(); if (shake && !reduced && !paused) ctx.translate(rand(-shake, shake), rand(-shake, shake));
    shake = Math.max(0, shake - visualDt * 40);
    background();
    if (game.state !== 'menu') {
      ctx.save(); ctx.strokeStyle = game.spreadTime > 0 ? '#b5ffda' : '#d2c0ff'; ctx.shadowColor = ctx.strokeStyle; ctx.shadowBlur = reduced ? 0 : 12; ctx.lineWidth = 2.5; ctx.lineCap = 'round';
      for (const b of game.bullets) { ctx.beginPath(); ctx.moveTo(b.x, b.y); ctx.lineTo(b.x - b.vx * .018, b.y + 12); ctx.stroke(); } ctx.restore();
      for (const e of game.enemies) { const color = e.kind === 'heavy' ? '#ff986e' : e.kind === 'shooter' ? '#ff7baf' : '#ad89ff'; const sides = e.kind === 'heavy' ? 6 : e.kind === 'shooter' ? 3 : 4; polygon(e.x, e.y, e.r, sides, e.rotation, color, `${color}18`); polygon(e.x, e.y, e.r * .45, sides, -e.rotation, color, `${color}10`); }
      for (const b of game.hostile) polygon(b.x, b.y, b.r, 4, clock, '#ff846e', '#ff846e80');
      for (const gem of game.gems) polygon(gem.x, gem.y, 6, 4, 0, '#b5ffda', '#b5ffda45');
      for (const item of game.pickups) {
        const color = { spread: '#d1a6ff', shield: '#81dfff', heal: '#b5ffda' }[item.kind];
        if (item.kind === 'heal') {
          ctx.save(); ctx.strokeStyle = '#b5ffda55'; ctx.lineWidth = 1;
          ctx.beginPath(); ctx.arc(item.x, item.y, reduced ? 25 : 25 + Math.sin(clock * 4) * 3, 0, Math.PI * 2); ctx.stroke(); ctx.restore();
          polygon(item.x, item.y, 21, 4, Math.PI / 4, color, '#15372c');
          ctx.save(); ctx.fillStyle = color;
          ctx.fillRect(item.x - 3, item.y - 9, 6, 18); ctx.fillRect(item.x - 9, item.y - 3, 18, 6);
          ctx.font = 'bold 9px sans-serif'; ctx.textAlign = 'center'; ctx.fillText('HP +1', item.x, item.y + 35); ctx.restore();
        } else {
          polygon(item.x, item.y, 14, 6, 0, color, '#16162a'); ctx.save(); ctx.fillStyle = color; ctx.font = 'bold 14px sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText({ spread: 'W', shield: 'S' }[item.kind], item.x, item.y); ctx.restore();
        }
      }
      if (game.boss) { const b = game.boss; polygon(b.x, b.y, b.r, 6, b.phase * .2, '#ff8c7f', '#421e3570'); polygon(b.x, b.y, b.r * .65, 6, -b.phase * .5, '#e9a7ff', '#b672ff20'); polygon(b.x, b.y, 12, 4, b.phase, '#fff0d2', '#ff866080'); polygon(b.x - 51, b.y + 10, 14, 3, -.4, '#ff8c7f', '#421e35'); polygon(b.x + 51, b.y + 10, 14, 3, .4, '#ff8c7f', '#421e35'); }
      if (game.state === 'playing' || paused) drawPlayer();
    }
    for (const p of particles) { p.life -= visualDt; p.x += p.vx * visualDt; p.y += p.vy * visualDt; p.vx *= Math.exp(-3 * visualDt); p.vy *= Math.exp(-3 * visualDt); ctx.globalAlpha = Math.max(0, p.life / p.max); ctx.fillStyle = p.color; ctx.fillRect(p.x, p.y, p.size, p.size); } ctx.globalAlpha = 1; particles = particles.filter(p => p.life > 0);
    for (const ring of rings) { ring.life -= visualDt; ring.r += visualDt * 145; ctx.globalAlpha = Math.max(0, ring.life / .45); ctx.strokeStyle = ring.color; ctx.lineWidth = 1; ctx.beginPath(); ctx.arc(ring.x, ring.y, ring.r, 0, Math.PI * 2); ctx.stroke(); } ctx.globalAlpha = 1; rings = rings.filter(r => r.life > 0);
    ctx.save(); ctx.font = '11px "Space Grotesk", sans-serif'; ctx.textAlign = 'center';
    for (const label of labels) { label.life -= visualDt; label.y -= visualDt * 30; ctx.globalAlpha = Math.max(0, label.life / .8); ctx.fillStyle = label.color; ctx.fillText(label.text, label.x, label.y); } ctx.restore(); labels = labels.filter(l => l.life > 0).slice(-35);
    if (flash > 0) { ctx.fillStyle = `rgba(255,100,80,${flash})`; ctx.fillRect(0, 0, width, height); flash = Math.max(0, flash - visualDt); } ctx.restore();
  }
  function frame(now) {
    const dt = last ? Math.min((now - last) / 1000, .1) : 0; last = now;
    if (game.state !== 'paused') { clock += dt; toastTime -= dt; if (toastTime <= 0) $('toast').classList.remove('show'); }
    if (game.state === 'playing') {
      accumulator += dt;
      while (accumulator >= 1 / 120 && game.state === 'playing') { game.update(1 / 120); accumulator -= 1 / 120; handleEvents(); }
      hudClock += dt; if (hudClock >= .08) { updateHUD(); hudClock = 0; }
    } else accumulator = 0;
    render(dt); requestAnimationFrame(frame);
  }
  bindMenu(); requestAnimationFrame(frame);
})();
