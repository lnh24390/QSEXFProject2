using System;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Windows.Forms;

namespace NeonStrikeLauncher
{
    static class Program
    {
        [STAThread]
        static void Main()
        {
            try
            {
                string gameDir = Path.Combine(
                    Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                    "NeonStrikeGame");
                Directory.CreateDirectory(gameDir);

                Extract("index.html", Path.Combine(gameDir, "index.html"));
                Extract("styles.css", Path.Combine(gameDir, "styles.css"));
                Extract("game.js", Path.Combine(gameDir, "game.js"));
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

                string page = new Uri(Path.Combine(gameDir, "index.html")).AbsoluteUri;
                ProcessStartInfo start = new ProcessStartInfo();
                start.FileName = edge;
                start.Arguments = "--app=\"" + page + "\" --app-id=NeonStrikeFPS --start-fullscreen --start-maximized --no-first-run";
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
