import urllib.request,pathlib,hashlib,zipfile
root=pathlib.Path(__file__).resolve().parents[2]/'android-build-cache'
root.mkdir(parents=True,exist_ok=True)
for name,sha,dest in [('build-tools_r35_windows.zip','af059bb67cf7786f45ee0db85e2d24985df1b4b6','tools'),('platform-35_r02.zip','0bb560a90a7a2cbd0dd8348224d518b638fe7949','platform')]:
 p=root/('verified-'+name)
 with urllib.request.urlopen('https://dl.google.com/android/repository/'+name,timeout=30) as r,p.open('wb') as f:
  while True:
   b=r.read(1024*1024)
   if not b:break
   f.write(b)
 assert hashlib.sha1(p.read_bytes()).hexdigest()==sha
 with zipfile.ZipFile(p) as z:z.extractall(root/dest)
 print(name+' verified and extracted',flush=True)
