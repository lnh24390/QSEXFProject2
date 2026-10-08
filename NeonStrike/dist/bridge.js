(async()=>{
  const ADDRESS='http://127.0.0.1:8765/';
  function openServerPicker(){const panel=document.createElement('section');panel.className='server-connect';const title=document.createElement('h2'),help=document.createElement('p'),input=document.createElement('input'),connect=document.createElement('button'),cancel=document.createElement('button'),error=document.createElement('p');title.textContent='다른 컴퓨터 서버 접속';help.textContent='서버 컴퓨터에서 EXE를 실행한 뒤 표시되는 접속 주소를 입력하세요. 모든 플레이어가 같은 서버에 접속해야 합니다.';input.placeholder='http://192.168.0.10:8765/';input.value=localStorage.getItem('neon-strike-last-server')||'';input.type='url';input.setAttribute('aria-label','게임 서버 주소');connect.textContent='서버 접속';cancel.textContent='닫기';connect.onclick=()=>{try{const value=input.value.trim();if(!value)throw Error();const url=new URL(value.includes('://')?value:'http://'+value);if(!['http:','https:'].includes(url.protocol)||url.username||url.password)throw Error();if(!url.port&&url.protocol==='http:')url.port='8765';url.pathname='/';url.search='';url.hash='';localStorage.setItem('neon-strike-last-server',url.href);location.assign(url.href)}catch{error.textContent='올바른 서버 주소를 입력하세요. 예: http://192.168.0.10:8765/'}};cancel.onclick=()=>panel.remove();input.onkeydown=e=>{if(e.key==='Enter')connect.click()};panel.append(title,help,input,connect,cancel,error);document.body.appendChild(panel);input.focus()}
  function connectionError(message){let box=document.getElementById('serverConnectionError');if(!box){box=document.createElement('section');box.id='serverConnectionError';box.style.cssText='position:fixed;inset:0;z-index:100;background:#06121c;color:#d9f7ff;display:grid;place-content:center;padding:30px;font:16px Arial;text-align:center';document.body.appendChild(box)}box.replaceChildren();const title=document.createElement('h2'),text=document.createElement('p'),link=document.createElement('a');title.textContent='NEON STRIKE 서버 연결';text.textContent=message;link.href=ADDRESS;link.textContent='서버로 접속 / 다시 연결';link.style.color='#28e7ff';const remote=document.createElement("button");remote.textContent="다른 컴퓨터 서버 접속";remote.onclick=openServerPicker;box.append(title,text,link,remote)}
  if(location.protocol==='file:'){try{const response=await fetch(ADDRESS+'api/health',{signal:AbortSignal.timeout(3000)});const health=await response.json();if(health.service!=='neonstrike-server-v1')throw Error();location.replace(ADDRESS);return}catch{connectionError('NeonStrike.exe를 먼저 실행해 서버를 켠 다음 아래 주소로 접속하세요.');return}}
  const base=location.origin;
  function rpc(path,body){const request=new XMLHttpRequest();request.open(body===undefined?'GET':'POST',base+path,false);if(body!==undefined)request.setRequestHeader('Content-Type','application/json');request.send(body===undefined?null:JSON.stringify(body));if(request.status!==200)throw Error('Game server unavailable');return JSON.parse(request.responseText)}
  try{if(rpc('/api/health').service!=='neonstrike-server-v1')throw Error()}catch{connectionError('공통 게임 서버가 필요합니다. NeonStrike.exe를 실행한 뒤 http://127.0.0.1:8765/ 로 접속하세요.');return}
  const shareInput=document.getElementById('serverShareLink'),copyStatus=document.getElementById('serverCopyStatus');
  shareInput.value=base+'/';
  shareInput.addEventListener('click',()=>shareInput.select());
  async function copyServerLink(){
    const value=shareInput.value;
    try{if(!navigator.clipboard?.writeText)throw Error();await navigator.clipboard.writeText(value);copyStatus.textContent='서버 링크가 복사되었습니다.'}
    catch{const previous=document.activeElement;shareInput.focus();shareInput.select();let copied=false;try{copied=document.execCommand('copy')}catch{}copyStatus.textContent=copied?'서버 링크가 복사되었습니다.':'주소가 선택되었습니다. Ctrl+C를 눌러 복사하세요.';if(copied)previous?.focus()}
  }
  document.getElementById('copyServerLinkBtn').onclick=copyServerLink;
  document.addEventListener('keydown',event=>{
    if(!(event.ctrlKey||event.metaKey)||event.key.toLowerCase()!=='c'||event.altKey||event.shiftKey)return;
    const target=event.target;
    if(target.closest?.('input,textarea,[contenteditable="true"]')||window.getSelection()?.toString())return;
    if(document.getElementById('menu').classList.contains('hidden')||document.querySelector('.server-connect'))return;
    event.preventDefault();copyServerLink();
  });
  document.getElementById('serverAddress').textContent=base;document.getElementById('changeServerBtn').onclick=openServerPicker;
  fetch(base+'/api/server-info').then(r=>r.json()).then(info=>{
    const local=['127.0.0.1','localhost','[::1]'].includes(location.hostname);
    const addresses=local&&info.addresses.length?info.addresses:[base+'/'];
    shareInput.value=addresses[0];
    document.getElementById('serverShareAddress').textContent='친구에게 공유할 주소: '+addresses.join(' 또는 ');
  }).catch(()=>{});
  const nativeStorage=window.localStorage,shared=key=>key==='neon-strike-rooms-v1'||key.startsWith('neon-strike-online-v1-');let keyCache=[],keyAt=0;
  function storageKeys(){if(Date.now()-keyAt>150){const local=[];for(let i=0;i<nativeStorage.length;i++){const key=nativeStorage.key(i);if(!shared(key))local.push(key)}keyCache=[...local,...rpc('/api/keys')];keyAt=Date.now()}return keyCache}
  window.neonStorage={getItem(key){return shared(key)?rpc('/api/store?key='+encodeURIComponent(key)):nativeStorage.getItem(key)},setItem(key,value){if(shared(key)){rpc('/api/store?key='+encodeURIComponent(key),{value:String(value)});keyAt=0}else nativeStorage.setItem(key,value)},removeItem(key){if(shared(key)){rpc('/api/store?key='+encodeURIComponent(key),{value:null});keyAt=0}else nativeStorage.removeItem(key)},get length(){return storageKeys().length},key(index){return storageKeys()[index]??null}};
  class GameServerChannel {
    constructor(topic){this.topic=topic;this.sender=Math.random().toString(36).slice(2)+Date.now().toString(36);this.onmessage=null;this.closed=false;this.queue=Promise.resolve();this.cursor=rpc('/api/channel?topic='+encodeURIComponent(topic)).cursor;this.timer=setInterval(()=>this.poll(),150);this.busy=false}
    postMessage(data){if(this.closed)return;const payload=JSON.parse(JSON.stringify(data));this.queue=this.queue.then(async()=>{const response=await fetch(base+'/api/channel?topic='+encodeURIComponent(this.topic),{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({sender:this.sender,data:payload})});if(!response.ok)throw Error('Game message failed')}).catch(()=>connectionError('서버 연결이 끊겼습니다. EXE를 실행하고 다시 연결하세요.'))}
    async poll(){if(this.closed||this.busy)return;this.busy=true;try{const response=await fetch(base+'/api/channel?topic='+encodeURIComponent(this.topic)+'&after='+this.cursor);if(!response.ok)throw Error();const result=await response.json();if(this.closed)return;this.cursor=result.cursor;for(const packet of result.events)if(packet.sender!==this.sender&&this.onmessage)this.onmessage({data:packet.data})}catch{connectionError('서버 연결이 끊겼습니다. EXE를 실행하고 다시 연결하세요.')}finally{this.busy=false}}
    close(){this.closed=true;clearInterval(this.timer);this.onmessage=null}
  }
  window.BroadcastChannel=GameServerChannel;window.NeonStrikeServerConnected=true;
  const game=document.createElement('script');game.src='game.js';document.body.appendChild(game);
})();
