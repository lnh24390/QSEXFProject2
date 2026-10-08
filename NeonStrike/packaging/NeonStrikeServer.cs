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
        static readonly Dictionary<string,string> Store = new Dictionary<string,string>();
        static readonly Dictionary<string,List<Packet>> Topics = new Dictionary<string,List<Packet>>();
        static readonly Dictionary<string,long> Cursors = new Dictionary<string,long>();
        sealed class Packet { public long seq; public string sender; public object data; }
        static string Root; static int Port;
        public static void Run(string root, int port) {
            Root=Path.GetFullPath(root); Port=port;
            TcpListener listener=new TcpListener(IPAddress.Any,port);
            try { listener.Start(); } catch(SocketException) { return; }
            while(true) { TcpClient client=listener.AcceptTcpClient(); ThreadPool.QueueUserWorkItem(delegate { Handle(client); }); }
        }
        static string ReadLine(Stream stream) {
            MemoryStream bytes=new MemoryStream(); int current;
            while((current=stream.ReadByte())!=-1) { if(current==10) break; if(current!=13) bytes.WriteByte((byte)current); if(bytes.Length>16384) throw new InvalidDataException(); }
            return Encoding.ASCII.GetString(bytes.ToArray());
        }
        static void Handle(TcpClient client) {
            using(client) { try {
                client.ReceiveTimeout=5000; client.SendTimeout=5000;
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
                if(health) output=new { service="neonstrike-server-v1" };
                else if(path=="/api/server-info"&&method=="GET") { List<string> addresses=new List<string>(); foreach(IPAddress address in Dns.GetHostAddresses(Dns.GetHostName())) if(address.AddressFamily==AddressFamily.InterNetwork&&!IPAddress.IsLoopback(address)) addresses.Add("http://"+address+":"+Port+"/"); output=new { port=Port,addresses=addresses }; }
                else if(path=="/api/keys"&&method=="GET") { lock(Gate) output=new List<string>(Store.Keys); }
                else if(path=="/api/store") { string key=Get(query,"key"); if(key!="neon-strike-rooms-v1"&&!key.StartsWith("neon-strike-online-v1-")) { Reply(stream,403,"text/plain",Encoding.UTF8.GetBytes("Key denied"),false); return; } lock(Gate) { if(method=="POST") { object value; if(!input.TryGetValue("value",out value)||value==null) Store.Remove(key); else Store[key]=Convert.ToString(value); } string saved; Store.TryGetValue(key,out saved); output=saved; } }
                else if(path=="/api/channel") { string topic=Get(query,"topic"); if(!topic.StartsWith("neon-strike-")) { Reply(stream,403,"text/plain",Encoding.UTF8.GetBytes("Topic denied"),false); return; } lock(Gate) { if(!Topics.ContainsKey(topic)) { Topics[topic]=new List<Packet>(); Cursors[topic]=0; } if(method=="POST") { object data,sender; if(!input.TryGetValue("data",out data)||!input.TryGetValue("sender",out sender)) throw new InvalidDataException(); long seq=++Cursors[topic]; Topics[topic].Add(new Packet { seq=seq,sender=Convert.ToString(sender),data=data }); if(Topics[topic].Count>4096) Topics[topic].RemoveRange(0,Topics[topic].Count-4096); output=new { cursor=seq }; } else { string after=Get(query,"after"); long cursor; if(after==""||!Int64.TryParse(after,out cursor)) output=new { cursor=Cursors[topic],events=new Packet[0] }; else { List<Packet> pending=Topics[topic].FindAll(p=>p.seq>cursor);if(pending.Count>128)pending=pending.GetRange(0,128);output=new { cursor=pending.Count>0?pending[pending.Count-1].seq:Cursors[topic],events=pending.ToArray() }; } } } }
                else if(path.StartsWith("/api/")) { Reply(stream,404,"text/plain",Encoding.UTF8.GetBytes("Not found"),false); return; }
                else { if(method!="GET") { Reply(stream,405,"text/plain",new byte[0],false); return; } string file=path=="/"?"index.html":path.TrimStart('/'); if(file!="index.html"&&file!="styles.css"&&file!="game.js"&&file!="bridge.js"&&file!="NeonStrike.ico") { Reply(stream,404,"text/plain",new byte[0],false); return; } string target=Path.Combine(Root,file); if(!File.Exists(target)) { Reply(stream,404,"text/plain",new byte[0],false); return; } string mime=file.EndsWith(".html")?"text/html; charset=utf-8":file.EndsWith(".css")?"text/css; charset=utf-8":file.EndsWith(".js")?"application/javascript; charset=utf-8":"image/x-icon"; Reply(stream,200,mime,File.ReadAllBytes(target),false); return; }
                Reply(stream,200,"application/json; charset=utf-8",Encoding.UTF8.GetBytes(json.Serialize(output)),health);
            } catch { try { Reply(client.GetStream(),400,"text/plain",Encoding.UTF8.GetBytes("Bad request"),false); } catch {} } }
        }
        static Dictionary<string,string> Query(string query) { Dictionary<string,string> values=new Dictionary<string,string>(); foreach(string part in query.TrimStart('?').Split('&')) { int equal=part.IndexOf('='); if(equal>=0) values[Uri.UnescapeDataString(part.Substring(0,equal))]=Uri.UnescapeDataString(part.Substring(equal+1)); } return values; }
        static string Get(Dictionary<string,string> values,string key) { string value; return values.TryGetValue(key,out value)?value:""; }
        static void Reply(Stream stream,int status,string mime,byte[] body,bool cors) { string header="HTTP/1.1 "+status+" "+(status==200?"OK":"Error")+"\r\nContent-Type: "+mime+"\r\nContent-Length: "+body.Length+"\r\nCache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\nConnection: close\r\n"+(cors?"Access-Control-Allow-Origin: *\r\n":"")+"\r\n"; byte[] bytes=Encoding.ASCII.GetBytes(header);stream.Write(bytes,0,bytes.Length);stream.Write(body,0,body.Length);stream.Flush(); }
    }
}
