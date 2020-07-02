/*
 * @Author: xieydd
 * @since: 2020-06-18 14:42:38
 * @lastTime: 2020-06-19 15:53:45
 * @LastAuthor: Do not edit
 * @message: 
 */
#include <string>
#include <stdio.h>
#include <fstream>
#include <vector>
#include <net.h>
#include <opencv2/opencv.hpp>

#define max(x, y) (x > y ? x : y)
#define min(x, y) (x < y ? x : y)

void print_usage(char **argv)
{
    fprintf(stderr, "./Extract image_dir outbin_dir model_dir threads\n");
    exit(0);
}

int save_mat2bin(std::string bin_filename, std::vector<ncnn::Mat> &outs, int width, int height)
{
    FILE *f = fopen(bin_filename.c_str(), "w+");
    fprintf(f, "%d %d \n", width, height);
    for (int i = 0; i < outs.size(); i++)
    {
        ncnn::Mat m = outs[i];
        for (int c = 0; c < m.c; c++)
        {
            const float *ptr = m.channel(c);
            for (int h = 0; h < m.h; h++)
            {
                for (int w = 0; w < m.w; w++)
                {
                    fprintf(f, "%f ", ptr[w]);
                }
                ptr += m.w;
            }
        }
        fprintf(f, "\n");
    }

    fclose(f);
    return 0;
}

int main(int argc, char **argv)
{
    std::string img_dir;
    std::string out_dir;
    std::string model_dir;
    ncnn::Net detector;

    int threads;
    std::vector<std::string> filenames;
    std::string val_txt;
    std::ifstream txt_file;
    int target_size = 320;
    int max_size = 320;
    float _mean_val[3] = {104.f, 117.f, 123.f};
    float _norm_val[3] = {1.f, 1.f, 1.f};

    if (argc != 5)
    {
        print_usage(argv);
        return -1;
    }

    img_dir = argv[1];
    out_dir = argv[2];
    model_dir = argv[3];
    sprintf(argv[4], "%d", &threads);

    std::string param = model_dir + "/slim_320_int8.param";
    std::string bin = model_dir + "/slim_320_int8.bin";

    detector.load_param(param.c_str());
    detector.load_model(bin.c_str());

    val_txt = img_dir.substr(0, img_dir.length() - 7) + "wider_val.txt";
    txt_file.open(val_txt, std::ios::in);
    if (!txt_file.is_open())
    {
        fprintf(stdout, "can`t open file %s\n", val_txt.c_str());
        return -1;
    }
    while (!txt_file.eof())
    {
        std::string sTmp;
        getline(txt_file, sTmp);
        filenames.push_back(img_dir + sTmp);
    }
    txt_file.close();

    for (int i = 0; i < filenames.size(); i++)
    {
        std::string filename = filenames[i];
        cv::Mat img = cv::imread(filename.c_str());
        int height = img.rows;
        int width = img.cols;
        int height_new, width_new;

        if (height > width)
        {
            height_new = target_size;
            width_new = int(1.0f*height/target_size)*width);
        }
        else
        {
            width_new = target_size;
            height_new = int(1.0f*width/target_size)*height);
        }

        ncnn::Mat in = ncnn::Mat::from_pixels_resize(img.data, ncnn::Mat::PIXEL_BGR, width, height, int(round(resize * width)), int(round(resize * height)));
        in.substract_mean_normalize(_mean_val, _norm_val);
        ncnn::Extractor ex = detector.create_extracor();
        ex.set_light_mode(true);
        ex.set_num_threads(threads);

        ex.input(0, in);
        ncnn::Mat loc, conf, landms;
        std::vector<ncnn::Mat> outs;
        ex.extract("362", loc);
        ex.extract("444", landms);
        ex.extract("445", conf);
        outs.push_back(loc);
        outs.push_back(conf);
        outs.push_back(landms);

        int index_start = filename.find_last_of('/');
        int index_end = filename.find_last_of('.');
        std::string bin_filename = out_dir + filename.substr(index_start, index_end - index_start) + ".bin";
        int ret = save_mat2bin(bin_filename, outs, width_new, height_new);
        if (ret != 0)
        {
            return -1;
        }
    }
    return 0;
}