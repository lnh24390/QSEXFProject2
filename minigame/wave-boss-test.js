(() => {
  'use strict';
  window.NEON_WAVE_TEST = true;
  const BaseGame = NeonCore.Game;
  const status = document.getElementById('test-status');
  let activeGame;
  function spawnTestBoss(game) {
    game.round = 10;
    game.bosses = game.mode === 'impossible' ? 18 : 9;
    game.spawnBoss();
    game.boss.y = 135;
    game.bullets = [];
    status.textContent = `각 색 체력 ${game.boss.waveMaxHP} · 9라운드 일반 보스 한 대와 동일`;
  }
  NeonCore.Game = class extends BaseGame {
    constructor(options) { super(options); activeGame = this; }
    reset(mode) {
      super.reset(mode);
      this.round = 10;
      this.bosses = this.mode === 'impossible' ? 18 : 9;
      this.nextBoss = 0;
      status.textContent = '시작하면 10라운드 웨이브 보스가 즉시 등장합니다.';
    }
    damage() {
      if (document.getElementById('test-invincible').checked) return false;
      return super.damage();
    }
  };
  document.querySelectorAll('[data-test-action]').forEach(button => {
    button.addEventListener('click', () => {
      const game = activeGame;
      if (!game || !['playing', 'paused'].includes(game.state)) {
        status.textContent = '먼저 게임 시작 또는 다시 도전을 누르세요.';
        return;
      }
      const action = button.dataset.testAction;
      if (action === 'restart') {
        game.pickups = [];
        spawnTestBoss(game);
      } else if (game.boss?.kind === 'wave') {
        const boss = game.boss;
        if (action === 'defeat') {
          boss.hp = 0;
          status.textContent = '보상은 5초 후 사라집니다. 다음 보스까지 30초.';
        } else if (action === 'clear') {
          boss.hp -= boss.waveHealth[boss.wave - 1];
          game.updateBossWave(boss);
          status.textContent = boss.hp ? `WAVE ${boss.wave} 시작` : '세 색 체력 소진 · 격파 보상을 확인하세요.';
        } else {
          const wave = Number(action);
          boss.hp = boss.waveMaxHP * (4 - wave);
          game.updateBossWave(boss);
          boss.shot = 0;
          game.hostile = []; game.bullets = [];
          status.textContent = `WAVE ${wave} 선택 · 각 색 최대 체력 ${boss.waveMaxHP}`;
        }
      } else status.textContent = '보스 다시 생성 버튼을 누르세요.';
      button.blur(); document.getElementById('game').focus();
    });
  });
})();
