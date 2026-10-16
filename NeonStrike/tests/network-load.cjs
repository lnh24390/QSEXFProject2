const assert = require('node:assert/strict');
const { spawn } = require('node:child_process');
const path = require('node:path');
const { setTimeout: delay } = require('node:timers/promises');
const http = require('node:http');
const port = Number(process.env.NEON_TEST_PORT || 18765);
const base = `http://127.0.0.1:${port}`;
const clientCount = Number(process.env.NEON_TEST_CLIENTS || 3);
assert(clientCount >= 2 && clientCount <= 10);
const executable = path.resolve(process.argv[2] || 'NeonStrike.exe');
let server;
async function reserveTestPort(){
  const net=require('node:net');
  await new Promise((resolve,reject)=>{const probe=net.createServer();probe.once('error',reject);probe.listen(port,'127.0.0.1',()=>probe.close(resolve))});
}
let failures = [], timings = [], completed = 0;
function request(url, body) {
  const begin = performance.now();
  return new Promise((resolve, reject) => {
    const data = body === undefined ? null : JSON.stringify(body);
    const req = http.request(base + url, { method: data ? 'POST' : 'GET', agent: false,
      headers: data ? { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(data) } : {} }, res => {
      let raw = ''; res.setEncoding('utf8'); res.on('data', chunk => raw += chunk);
      res.on('end', () => { timings.push(performance.now() - begin); completed++;
        if (res.statusCode !== 200) return reject(Error(`${url}: HTTP ${res.statusCode} ${raw.slice(0,80)}`));
        try { resolve(JSON.parse(raw)); } catch (err) { reject(err); }
      });
    });
    req.setTimeout(5000, () => req.destroy(Error('request timeout')));
    req.on('error', reject); if (data) req.write(data); req.end();
  });
}
(async () => {
 try {
  await reserveTestPort();
  server=spawn(executable,['--server',path.resolve('dist'),String(port)],{windowsHide:true,stdio:'ignore'});
  server.on('error',()=>{});
  for(let i=0;;i++){try{await request('/api/health');break}catch(e){if(i===50)throw e;await delay(100)}}
  const members = Array.from({length:clientCount},(_,i)=>({id:'load-'+i,name:'LOAD '+i,team:i%2?'red':'blue',weapon:'k2'}));
  await request('/api/store?key=neon-strike-rooms-v1',{value:JSON.stringify([{code:'LOAD',hostId:members[0].id,mode:'pvp',members,players:clientCount}])});
  for(const p of members) await request('/api/session',{...p,room:'LOAD',action:'join',sentAt:Date.now()});
  const draft = await request('/api/session',{id:members[0].id,room:'LOAD',action:'draft',map:'sector9'});
  for(const p of members) await request('/api/session',{id:p.id,room:'LOAD',action:'confirmWeapon',draftEndsAt:draft.match.draftEndsAt,weapon:'k2'});
  await delay(1100);
  const state = (i,n) => ({sender:members[i].id,data:{...members[i],type:'state',x:25+n/10000,y:65,a:1.1,hp:100,ammo:30,reserve:120,kills:0,assists:0,deaths:0,dead:false,reloadProgress:1,reloadActive:false,pitch:0}});
  // Three moving players plus overlapping session/channel reads. Higher-than-game
  // request rate makes races reproducible without depending on a slow network.
  await Promise.all(Array.from({length:18},async(_,worker)=>{
    for(let n=0;n<180;n++){
      try {
        if(worker%3===0) await request('/api/channel?topic=neon-strike-LOAD',state((worker+n)%clientCount,n));
        else if(worker%3===1) await request('/api/session',{id:members[n%clientCount].id,room:'LOAD',action:'ping',sentAt:Date.now()});
        else await request('/api/channel?topic=neon-strike-LOAD&after=0');
      } catch(e){failures.push(e.message)}
    }
  }));
  const result=await request('/api/session',{id:members[0].id,room:'LOAD',action:'ping'});
  assert.equal(result.match.stage,'running');
  assert.equal(result.players.length,clientCount);
  timings.sort((a,b)=>a-b);
  console.log(JSON.stringify({clients:clientCount,completed,failures:failures.length,examples:failures.slice(0,5),p95Ms:Math.round(timings[Math.floor(timings.length*.95)]),maxMs:Math.round(timings.at(-1))}));
  assert.equal(failures.length,0,'Concurrent requests must all succeed');
 } finally { server?.kill(); }
})().catch(e=>{console.error(e.message);process.exitCode=1});
