using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using OpenCvSharp;
using OpenCvSharp.Dnn;

namespace ObjectClassification
{
    class Program
    {
        static void Main(string[] args)
        {
                const string prototext = @"..\..\..\..\data\bvlc_googlenet.prototxt";
                const string caffeModel = @"..\..\..\..\data\bvlc_googlenet.caffemodel";
                const string synsetWords = @"..\..\..\..\data\synset_words.txt";


                string[] classNames = File.ReadAllLines(synsetWords).Select(l => l.Split(' ').Last()).ToArray();
                //Use stopwatch object fro timing of the operation
                Stopwatch sw = new Stopwatch();

                string imgPath = @"D:\DeepLearningOpenCV\images\DogBycleCar.jpg";



                using (var net = CvDnn.ReadNetFromCaffe(prototext, caffeModel))
                using (var img = Cv2.ImRead(imgPath))
                {
                    //Just out of curiosity, I wanted to get the  Layer names of the NN Construct
                    // by calling GetLayerNames method of the Net object
                    string[] layerNames = net.GetLayerNames();
                    Console.WriteLine("Layer names : {0}", string.Join(", ", layerNames));
                    Console.WriteLine();

                    using (var inputBlob = CvDnn.BlobFromImage(img, 1, new Size(224, 224), new Scalar(104, 117, 123), swapRB: true, crop: false))
                    {
                        sw.Start();
                        net.SetInput(inputBlob, "data");
                        using (var prob = net.Forward("prob"))
                        {
                            sw.Stop();
                            Console.WriteLine($"Cost of calculating prob {sw.ElapsedMilliseconds} ms");
                            int cols = prob.Cols;
                            int rows = prob.Rows;
                            Console.WriteLine("Cols: " + cols + ", Rows:" + rows);
                            // GetMaxProClass(prob, out int classId, out double classProb);
                            Cv2.MinMaxLoc(prob, out _, out double classProb, out _, out Point classNumberPoint);
                            int classId = classNumberPoint.X;


                            Console.WriteLine("Best class: #{0}, '{1}'", classId, classNames[classId]);

                            Console.WriteLine("Probability:{0:P2}", classProb);
                            string txt = "Label: " + classNames[classId] + ", % " + (100 * classProb).ToString("0.####");
                            Cv2.PutText(img, txt, new Point(5, 25), HersheyFonts.HersheySimplex, 0.7, new Scalar(0, 0, 255), 2);
                            //Cv2.ImWrite("classification.jpg", img);
                            Cv2.ImShow("image", img);
                        }
                    }
                }
                Cv2.WaitKey();
                Cv2.DestroyAllWindows();

                //  Console.Write("Downloading Caffe Model...");
                Console.WriteLine("Press any key to exit");
                Console.Read();
            }
        }
    }
    

