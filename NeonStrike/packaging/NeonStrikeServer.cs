using System;
using System.Collections.Generic;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Web.Script.Serialization;
namespace NeonStrikeLauncher {
    static class LocalServer {
        static readonly object Gate = new object();
        static readonly string InstanceId = Guid.NewGuid().ToString("N");
        static readonly Dictionary<string,string> Store = new Dictionary<string,string>();
        static readonly Dictionary<string,List<Packet>> Topics = new Dictionary<string,List<Packet>>();
        static readonly Dictionary<string,long> Cursors = new Dictionary<string,long>();
        sealed class Packet { public long seq; public string sender; public object data; }
        static string Root; static int Port; static Timer MatchTimer;
        sealed class Session { public string id,room; public long seen; public bool connected=true; }
        sealed class Match { public string stage="draft",map="sector9",winner,surrenderedTeam; public long startedAt,finishedAt,draftEndsAt; public int blue,red; public bool desertion; public List<Dictionary<string,object>> roster=new List<Dictionary<string,object>>(); public Dictionary<string,object> states=new Dictionary<string,object>(); }
        static readonly Dictionary<string,Session> Sessions=new Dictionary<string,Session>();
        static readonly Dictionary<string,Match> Matches=new Dictionary<string,Match>();
        static long Now() { return DateTimeOffset.UtcNow.ToUnixTimeMilliseconds(); }
        static string Str(Dictionary<string,object> d,string k) { object v; return d.TryGetValue(k,out v)?Convert.ToString(v):""; }
        static List<Dictionary<string,object>> Rooms() { string s; return Store.TryGetValue("neon-strike-rooms-v1",out s)?new JavaScriptSerializer().Deserialize<List<Dictionary<string,object>>>(s):new List<Dictionary<string,object>>(); }
        static string NormalizeRoomStore(string value) {
            var json=new JavaScriptSerializer();var rooms=json.Deserialize<List<Dictionary<string,object>>>(value);if(rooms==null)return "[]";
            foreach(var room in rooms) {object raw;if(!room.TryGetValue("members",out raw))continue;var members=json.Deserialize<List<Dictionary<string,object>>>(json.Serialize(raw));var ids=new HashSet<string>();var unique=new List<Dictionary<string,object>>();if(members!=null)foreach(var member in members) {string id=Str(member,"id");if(id.Length>0&&ids.Add(id))unique.Add(member);}room["members"]=unique;room["players"]=unique.Count;}
            return json.Serialize(rooms);
        }
        static void Emit(string room,object data) { string topic="neon-strike-"+room;if(!Topics.ContainsKey(topic)){Topics[topic]=new List<Packet>();Cursors[topic]=0;}Topics[topic].Add(new Packet{seq=++Cursors[topic],sender="server",data=data}); }
        static void AbortDraft(string room,Match m) { m.stage="lobby"; Emit(room,new {type="draftabort",id="server"}); }
        static void Observe(string topic,Dictionary<string,object> d) {
            string room=topic.Substring("neon-strike-".Length),type=Str(d,"type"),pid=Str(d,"id"); Match m;Matches.TryGetValue(room,out m);
            if(type=="countdown"&&(m==null||m.stage!="draft"&&m.stage!="running")) { var r=Rooms().Find(x=>Str(x,"code")==room);if(r==null||Str(r,"hostId")!=pid)return;m=new Match();m.map=Str(d,"map");object deadline;if(d.TryGetValue("endsAt",out deadline))m.draftEndsAt=Convert.ToInt64(deadline);object members;if(r.TryGetValue("members",out members))m.roster=new JavaScriptSerializer().Deserialize<List<Dictionary<string,object>>>(new JavaScriptSerializer().Serialize(members));Matches[room]=m; }
            if(m==null)return;
            if(type=="surrenderState"&&m.stage=="running") {
                var roomInfo=Rooms().Find(x=>Str(x,"code")==room);object rawVote;
                if(roomInfo!=null&&Str(roomInfo,"hostId")==pid&&d.TryGetValue("vote",out rawVote)) {
                    var vote=rawVote as Dictionary<string,object>;object rawMembers,rawBallots;
                    if(vote!=null&&Str(vote,"result")=="passed"&&(Str(vote,"side")=="blue"||Str(vote,"side")=="red")&&vote.TryGetValue("members",out rawMembers)&&vote.TryGetValue("ballots",out rawBallots)) {
                        var serializer=new JavaScriptSerializer();var members=serializer.Deserialize<List<string>>(serializer.Serialize(rawMembers));var ballots=rawBallots as Dictionary<string,object>;var unique=new HashSet<string>(members);int yes=0;bool valid=members.Count>0&&unique.Count==members.Count;
                        foreach(string member in members) {if(!m.roster.Exists(p=>Str(p,"id")==member&&Str(p,"team")==Str(vote,"side")))valid=false;object choice;if(ballots!=null&&ballots.TryGetValue(member,out choice)&&Convert.ToString(choice)=="yes")yes++;}
                        if(valid&&yes>=members.Count/2+1) {m.stage="ended";m.finishedAt=Now();m.desertion=false;m.surrenderedTeam=Str(vote,"side");m.winner=m.surrenderedTeam=="blue"?"RED":"BLUE";Emit(room,new{type="matchend",id="server",winner=m.winner,blue=m.blue,red=m.red,surrenderedTeam=m.surrenderedTeam,desertion=false});}
                    }
                }
            }

            if(type=="matchstart"&&m.stage=="draft") { m.stage="running";m.map=Str(d,"map");m.startedAt=Now()+3000; }
            if(type=="countdowncancel"&&m.stage=="draft")m.stage="lobby";
            if(type=="state"||type=="join"||type=="death") { d["stateAt"]=Now();m.states[pid]=new Dictionary<string,object>(d); }
            if(type=="score"||type=="coopBots"||type=="matchend") {object v;if(d.TryGetValue("blue",out v))m.blue=Convert.ToInt32(v);if(d.TryGetValue("red",out v))m.red=Convert.ToInt32(v);}
            if(type=="matchend") {m.stage="ended";m.winner=Str(d,"winner");m.surrenderedTeam=Str(d,"surrenderedTeam");m.finishedAt=Now();object reason;m.desertion=d.TryGetValue("desertion",out reason)&&reason is bool&&(bool)reason;}
            if(type=="leave") {Session s;if(Sessions.TryGetValue(pid,out s)){s.connected=false;s.seen=Now();}if(m.stage=="draft")AbortDraft(room,m);}
        }
        static bool Online(string pid,long now) { Session s;return Sessions.TryGetValue(pid,out s)&&s.connected&&now-s.seen<30000; }
        static bool Gone(string pid,long now) {Session s;return !Sessions.TryGetValue(pid,out s)||now-s.seen>=30000;}
        static void Tick() { lock(Gate) { long now=Now();var rooms=Rooms();bool dirty=false;
            foreach(var r in rooms) {string code=Str(r,"code");Match m;Matches.TryGetValue(code,out m);object raw;if(!r.TryGetValue("members",out raw))continue;var members=new JavaScriptSerializer().Deserialize<List<Dictionary<string,object>>>(new JavaScriptSerializer().Serialize(raw));
                var removed=members.RemoveAll(p=>{Session s;return Sessions.TryGetValue(Str(p,"id"),out s)&&(!s.connected||now-s.seen>=30000);});
                if(removed>0) {dirty=true;r["members"]=members;r["players"]=members.Count;if(m!=null&&m.stage=="draft")AbortDraft(code,m);}
                if(m!=null&&m.stage=="draft"&&m.draftEndsAt>0&&now>=m.draftEndsAt){m.stage="running";m.startedAt=now+3000;r["inMatch"]=true;dirty=true;Emit(code,new{type="matchstart",id="server",map=m.map});}
                if(m!=null&&m.stage=="running") {bool bg=m.roster.Exists(p=>Str(p,"team")=="blue")&&m.roster.FindAll(p=>Str(p,"team")=="blue").TrueForAll(p=>Gone(Str(p,"id"),now));bool rg=Str(r,"mode")!="coop"&&m.roster.Exists(p=>Str(p,"team")=="red")&&m.roster.FindAll(p=>Str(p,"team")=="red").TrueForAll(p=>Gone(Str(p,"id"),now));
                    if(bg||rg||now>=m.startedAt+1500000) {m.desertion=bg||rg;m.stage="ended";m.finishedAt=now;m.winner=bg&&rg?"DRAW":bg?"RED":rg?"BLUE":m.blue==m.red?"DRAW":m.blue>m.red?"BLUE":"RED";m.surrenderedTeam=bg&&!rg?"blue":rg&&!bg?"red":null;Emit(code,new{type="matchend",id="server",winner=m.winner,blue=m.blue,red=m.red,surrenderedTeam=m.surrenderedTeam,desertion=bg||rg});}
                }
                if(m!=null&&m.stage=="ended"&&now-m.finishedAt>=3000) {r["inMatch"]=false;dirty=true;}
                if(members.Count>0&&!members.Exists(p=>Str(p,"id")==Str(r,"hostId"))) {r["hostId"]=Str(members[0],"id");r["host"]=Str(members[0],"name");dirty=true;}
                if(members.Count>0||m!=null&&m.stage=="running"){r["updated"]=now;dirty=true;}
            }
            rooms.RemoveAll(r=>{object n;Match m;Matches.TryGetValue(Str(r,"code"),out m);return r.TryGetValue("players",out n)&&Convert.ToInt32(n)==0&&(m==null||m.stage!="running");});
            if(dirty)Store["neon-strike-rooms-v1"]=new JavaScriptSerializer().Serialize(rooms);
        } }
        public static void Run(string root, int port) {
            Root=Path.GetFullPath(root); Port=port;
            TcpListener listener=new TcpListener(IPAddress.Any,port);
            try { listener.Start(); } catch(SocketException) { return; }
            int workers,io,maxWorkers,maxIo;ThreadPool.GetMinThreads(out workers,out io);ThreadPool.GetMaxThreads(out maxWorkers,out maxIo);ThreadPool.SetMinThreads(Math.Max(workers,Math.Min(32,maxWorkers)),io);
            MatchTimer=new Timer(delegate { try { Tick(); } catch {} },null,1000,1000);
            while(true) { TcpClient client=listener.AcceptTcpClient(); ThreadPool.QueueUserWorkItem(delegate { Handle(client); }); }
        }
        static string ReadLine(Stream stream) {
            MemoryStream bytes=new MemoryStream(); int current;
            while((current=stream.ReadByte())!=-1) { if(current==10) break; if(current!=13) bytes.WriteByte((byte)current); if(bytes.Length>16384) throw new InvalidDataException(); }
            return Encoding.ASCII.GetString(bytes.ToArray());
        }
        static void Handle(TcpClient client) {
            using(client) { try {
                client.NoDelay=true; client.ReceiveTimeout=5000; client.SendTimeout=5000;
                NetworkStream stream=client.GetStream(); string first=ReadLine(stream); string[] parts=first.Split(' ');
                if(parts.Length<2) return; string method=parts[0]; Uri uri=new Uri("http://127.0.0.1:"+Port+parts[1]);
                int length=0,totalHeaders=0; string origin="",host=""; string line;
                while((line=ReadLine(stream)).Length>0) { totalHeaders+=line.Length; if(totalHeaders>32768) throw new InvalidDataException(); int colon=line.IndexOf(':'); if(colon<0) continue; string key=line.Substring(0,colon).Trim().ToLowerInvariant(),value=line.Substring(colon+1).Trim(); if(key=="content-length") Int32.TryParse(value,out length); if(key=="origin") origin=value; if(key=="host") host=value; }
                if(length<0||length>1048576) { Reply(stream,413,"text/plain",Encoding.UTF8.GetBytes("Request too large"),false); return; }
                bool health=uri.AbsolutePath=="/api/health";
                bool deniedOrigin=!health&&origin.Length>0&&origin!="http://"+host;
                if(method=="OPTIONS") { Reply(stream,200,"text/plain",new byte[0],health); return; }
                byte[] body=new byte[length]; int read=0; while(read<length) { int n=stream.Read(body,read,length-read); if(n<=0) return; read+=n; }
                if(deniedOrigin) { Reply(stream,403,"text/plain",Encoding.UTF8.GetBytes("Origin denied"),false); return; }
                JavaScriptSerializer json=new JavaScriptSerializer(); json.MaxJsonLength=4194304;
                Dictionary<string,object> input=length==0?new Dictionary<string,object>():json.Deserialize<Dictionary<string,object>>(Encoding.UTF8.GetString(body));
                Dictionary<string,string> query=Query(uri.Query); object output=null; string path=uri.AbsolutePath;
                if(health) output=new { service="neonstrike-server-v1",instanceId=InstanceId };
                else if(path=="/api/session"&&method=="POST") { lock(Gate) {string pid=Str(input,"id"),room=Str(input,"room");if(pid.Length<3||pid.Length>80||room.Length>32)throw new InvalidDataException();Session s;if(!Sessions.TryGetValue(pid,out s)){s=new Session{id=pid,room=room};Sessions[pid]=s;}s.room=room;s.seen=Now();s.connected=Str(input,"action")!="leave";Match m;Matches.TryGetValue(room,out m);if(Str(input,"action")=="draft") {var r=Rooms().Find(x=>Str(x,"code")==room);if(r!=null&&Str(r,"hostId")==pid&&(m==null||m.stage=="lobby"||m.stage=="ended")){m=new Match();m.map=Str(input,"map");m.draftEndsAt=Now()+60000;m.roster=json.Deserialize<List<Dictionary<string,object>>>(json.Serialize(r["members"]));Matches[room]=m;Emit(room,new{type="countdown",id="server",value=60,endsAt=m.draftEndsAt,map=m.map});}}if(!s.connected&&m!=null&&m.stage=="draft")AbortDraft(room,m);if(s.connected&&m!=null&&m.stage=="running"&&m.roster.Exists(p=>Str(p,"id")==pid)){var rooms=Rooms();var r=rooms.Find(x=>Str(x,"code")==room);if(r!=null){var members=json.Deserialize<List<Dictionary<string,object>>>(json.Serialize(r["members"]));if(!members.Exists(p=>Str(p,"id")==pid)){members.Add(m.roster.Find(p=>Str(p,"id")==pid));r["members"]=members;r["players"]=members.Count;r["updated"]=Now();Store["neon-strike-rooms-v1"]=json.Serialize(rooms);}}}var peers=new List<object>();if(m!=null&&m.stage=="running")foreach(var pair in m.states)if(Online(pair.Key,Now())&&m.roster.Exists(p=>Str(p,"id")==pair.Key))peers.Add(pair.Value);output=new {match=m,players=peers,serverTime=Now()}; } }
                else if(path=="/api/server-info"&&method=="GET") { List<string> addresses=new List<string>(); foreach(IPAddress address in Dns.GetHostAddresses(Dns.GetHostName())) if(address.AddressFamily==AddressFamily.InterNetwork&&!IPAddress.IsLoopback(address)) addresses.Add("http://"+address+":"+Port+"/"); output=new { port=Port,addresses=addresses }; }
                else if(path=="/api/shared"&&method=="GET") { lock(Gate) output=new Dictionary<string,string>(Store); }
                else if(path=="/api/keys"&&method=="GET") { lock(Gate) output=new List<string>(Store.Keys); }
                else if(path=="/api/store") { string key=Get(query,"key"); if(key!="neon-strike-rooms-v1"&&!key.StartsWith("neon-strike-online-v1-")) { Reply(stream,403,"text/plain",Encoding.UTF8.GetBytes("Key denied"),false); return; } lock(Gate) { if(method=="POST") { object value; if(!input.TryGetValue("value",out value)||value==null) Store.Remove(key); else Store[key]=key=="neon-strike-rooms-v1"?NormalizeRoomStore(Convert.ToString(value)):Convert.ToString(value); } string saved; Store.TryGetValue(key,out saved); output=saved; } }
                else if(path=="/api/channel") { string topic=Get(query,"topic"); if(!topic.StartsWith("neon-strike-")) { Reply(stream,403,"text/plain",Encoding.UTF8.GetBytes("Topic denied"),false); return; } lock(Gate) { if(!Topics.ContainsKey(topic)) { Topics[topic]=new List<Packet>(); Cursors[topic]=0; } if(method=="POST") { object data,sender; if(!input.TryGetValue("data",out data)||!input.TryGetValue("sender",out sender)) throw new InvalidDataException(); var eventData=data as Dictionary<string,object>;if(eventData!=null)Observe(topic,eventData); long seq=++Cursors[topic]; Topics[topic].Add(new Packet { seq=seq,sender=Convert.ToString(sender),data=data }); if(Topics[topic].Count>4096) Topics[topic].RemoveRange(0,Topics[topic].Count-4096); output=new { cursor=seq }; } else { string after=Get(query,"after"); long cursor; if(after==""||!Int64.TryParse(after,out cursor)) output=new { cursor=Cursors[topic],events=new Packet[0] }; else { List<Packet> pending=Topics[topic].FindAll(p=>p.seq>cursor);if(pending.Count>128)pending=pending.GetRange(0,128);output=new { cursor=pending.Count>0?pending[pending.Count-1].seq:Cursors[topic],events=pending.ToArray() }; } } } }
                else if(path.StartsWith("/api/")) { Reply(stream,404,"text/plain",Encoding.UTF8.GetBytes("Not found"),false); return; }
                else { if(method!="GET") { Reply(stream,405,"text/plain",new byte[0],false); return; } string file=path=="/"?"index.html":path.TrimStart('/'); if(file!="index.html"&&file!="styles.css"&&file!="game.js"&&file!="bridge.js"&&file!="NeonStrike.ico") { Reply(stream,404,"text/plain",new byte[0],false); return; } string target=Path.Combine(Root,file); if(!File.Exists(target)) { Reply(stream,404,"text/plain",new byte[0],false); return; } string mime=file.EndsWith(".html")?"text/html; charset=utf-8":file.EndsWith(".css")?"text/css; charset=utf-8":file.EndsWith(".js")?"application/javascript; charset=utf-8":"image/x-icon"; Reply(stream,200,mime,File.ReadAllBytes(target),false); return; }
                Reply(stream,200,"application/json; charset=utf-8",Encoding.UTF8.GetBytes(json.Serialize(output)),health);
            } catch { try { Reply(client.GetStream(),400,"text/plain",Encoding.UTF8.GetBytes("Bad request"),false); } catch {} } }
        }
        static Dictionary<string,string> Query(string query) { Dictionary<string,string> values=new Dictionary<string,string>(); foreach(string part in query.TrimStart('?').Split('&')) { int equal=part.IndexOf('='); if(equal>=0) values[Uri.UnescapeDataString(part.Substring(0,equal))]=Uri.UnescapeDataString(part.Substring(equal+1)); } return values; }
        static string Get(Dictionary<string,string> values,string key) { string value; return values.TryGetValue(key,out value)?value:""; }
        static void Reply(Stream stream,int status,string mime,byte[] body,bool cors) { string header="HTTP/1.1 "+status+" "+(status==200?"OK":"Error")+"\r\nContent-Type: "+mime+"\r\nContent-Length: "+body.Length+"\r\nCache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\nConnection: close\r\n"+(cors?"Access-Control-Allow-Origin: *\r\n":"")+"\r\n"; byte[] bytes=Encoding.ASCII.GetBytes(header);byte[] response=new byte[bytes.Length+body.Length];Buffer.BlockCopy(bytes,0,response,0,bytes.Length);Buffer.BlockCopy(body,0,response,bytes.Length,body.Length);stream.Write(response,0,response.Length);stream.Flush(); }
    }
}
