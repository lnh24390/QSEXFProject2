using System;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Net;
using System.Threading;
using System.Windows.Forms;

namespace NeonStrikeLauncher
{
    static class Program
    {
        [STAThread]
        static void Main(string[] args)
        {
            if (args.Length > 0 && args[0] == "--server") {
                string serverRoot = args.Length > 1 ? args[1] : Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "NeonStrikeGame");
                int port = args.Length > 2 ? Int32.Parse(args[2]) : 8765;
                LocalServer.Run(serverRoot, port); return;
            }
            try
            {
                string gameDir = Path.Combine(
                    Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                    "NeonStrikeGame");
                Directory.CreateDirectory(gameDir);

                Extract("index.html", Path.Combine(gameDir, "index.html"));
                Extract("styles.css", Path.Combine(gameDir, "styles.css"));
                Extract("game.js", Path.Combine(gameDir, "game.js"));
                Extract("bridge.js", Path.Combine(gameDir, "bridge.js"));
                Extract("NeonStrike.ico", Path.Combine(gameDir, "NeonStrike.ico"));

                string edge = FindEdge();
                if (edge == null)
                {
                    MessageBox.Show(
                        "Microsoft Edge is required to run NEON STRIKE.",
                        "NEON STRIKE",
                        MessageBoxButtons.OK,
                        MessageBoxIcon.Error);
                    return;
                }

                string page = "http://127.0.0.1:8765/";
                EnsureServer(gameDir);
                ProcessStartInfo start = new ProcessStartInfo();
                start.FileName = edge;
                string browserProfile = Path.Combine(gameDir, "BrowserProfile");
                Directory.CreateDirectory(browserProfile);
                start.Arguments = "--app=\"" + page + "\" --user-data-dir=\"" + browserProfile + "\" --start-fullscreen --start-maximized --no-first-run --no-default-browser-check";
                start.UseShellExecute = true;
                Process.Start(start);
            }
            catch (Exception ex)
            {
                MessageBox.Show(
                    "Failed to launch the game.\n\n" + ex.Message,
                    "NEON STRIKE",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Error);
            }
        }

        static bool ServerRunning() { try { HttpWebRequest request=(HttpWebRequest)WebRequest.Create("http://127.0.0.1:8765/api/health");request.Timeout=500;request.ReadWriteTimeout=500;using(WebResponse response=request.GetResponse())using(StreamReader reader=new StreamReader(response.GetResponseStream()))return reader.ReadToEnd().Contains("neonstrike-server-v1"); } catch { return false; } }
        static void EnsureServer(string gameDir) {
            if(ServerRunning()) return;
            ProcessStartInfo server=new ProcessStartInfo(Assembly.GetExecutingAssembly().Location);
            server.Arguments="--server \""+gameDir+"\""; server.UseShellExecute=false;server.CreateNoWindow=true;server.WindowStyle=ProcessWindowStyle.Hidden;
            Process.Start(server);
            for(int i=0;i<50;i++) { Thread.Sleep(100); if(ServerRunning())return; }
            throw new InvalidOperationException("Cannot start local game server on 127.0.0.1:8765. Check if the port is already in use.");
        }

        static void Extract(string resourceName, string outputPath)
        {
            Assembly assembly = Assembly.GetExecutingAssembly();
            using (Stream input = assembly.GetManifestResourceStream(resourceName))
            {
                if (input == null) throw new InvalidOperationException("Missing game resource: " + resourceName);
                using (FileStream output = new FileStream(outputPath, FileMode.Create, FileAccess.Write))
                {
                    input.CopyTo(output);
                }
            }
        }

        static string FindEdge()
        {
            string[] candidates = new string[]
            {
                Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ProgramFilesX86), "Microsoft", "Edge", "Application", "msedge.exe"),
                Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles), "Microsoft", "Edge", "Application", "msedge.exe")
            };
            foreach (string path in candidates)
            {
                if (File.Exists(path)) return path;
            }
            return null;
        }
    }
}
