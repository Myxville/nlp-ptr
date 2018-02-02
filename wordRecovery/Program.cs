using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices.ComTypes;
using System.Text.RegularExpressions;

namespace wordRecovery
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Welcome to the world recovery tool");
            Console.WriteLine("==================================");
            Console.WriteLine();
            Console.WriteLine("Seat back and relax, we are loading database...");
            var watch = System.Diagnostics.Stopwatch.StartNew();
            var text = File.ReadAllText(@".\Data\Harry Potter and the Deathly Hallows.txt");
            LanguageModel langModel = new LanguageModel();

            Regex rgx = new Regex("[^a-zA-Z ]");
            text = rgx.Replace(text, "");

            langModel.Initialize(text.ToLowerInvariant());

            MissingLetterModel missingModel = new MissingLetterModel();
            var allLetters = string.Join("", langModel.Vocabulary.ToArray());
            var substitutions = new String('-', allLetters.Length);

            var data = new List<KeyValuePair<string, string>>
            {
                new KeyValuePair<string, string>(allLetters, substitutions),
                new KeyValuePair<string, string>(allLetters, allLetters),
                new KeyValuePair<string, string>("aeiouy", "------"),
            };
            missingModel.Initialize(data);

            watch.Stop();
            Console.WriteLine("Yey, it took me only {0} ms to learn your language", watch.ElapsedMilliseconds);
            Console.WriteLine("Let the magic begin!");
            while (true)
            {
                Console.Write("Please enter string to recover: ");
                var toRecover = Console.ReadLine();

                if (string.IsNullOrEmpty(toRecover))
                    break;

                Console.WriteLine();

                if (toRecover != null)
                {
                    var candidates = Functions.NoisyChannel(toRecover.ToLowerInvariant(), langModel, missingModel);
                    foreach (var candidate in candidates)
                    {
                        Console.WriteLine("{0} {1}", candidate.Key, candidate.Value);
                    }
                }

                Console.WriteLine("===================");
            }

            Console.ReadLine();
        }
    }
}
