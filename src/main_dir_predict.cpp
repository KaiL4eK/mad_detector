#include <boost/program_options.hpp>
namespace po = boost::program_options;

#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;

#include <boost/algorithm/string/predicate.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/foreach.hpp>

#include <cmath>
#include <iostream>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>

using namespace std;

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include "yolo_ov.hpp"

#include <ros/ros.h>
#include <mad_detector/Detection.h>
#include <mad_detector/Detections.h>
#include <geometry_msgs/Point.h>

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>


string g_ir_path;
string g_cfg_path;
string g_device_type;
string g_dirpath;


int main(int argc, char **argv)
{
    ros::init(argc, argv, "yolo_detector");
    ros::NodeHandle pr_nh("~");
    ros::NodeHandle nh;

    pr_nh.getParam("config_path", g_cfg_path);
    pr_nh.getParam("ir_path", g_ir_path);
    pr_nh.getParam("device", g_device_type);
    pr_nh.getParam("dirpath", g_dirpath);

    fs::path input_dir(g_dirpath);
    fs::path output_dir = fs::path(g_dirpath) / fs::path("_predict");
    fs::recursive_directory_iterator iter(input_dir), eod;

    if ( fs::exists(output_dir) )
        fs::remove_all(output_dir);

    fs::create_directory(output_dir);

    vector<string> parse_list;

    BOOST_FOREACH(fs::path const& i, make_pair(iter, eod)){
        if (boost::ends_with(i.string(), ".png")){
            parse_list.push_back(i.string());
        }
    }

    YOLO_OpenVINO yolo(g_cfg_path);
    yolo.init(g_ir_path, g_device_type);

    vector<string> labels = yolo.get_labels();

    size_t total_frames = labels.size();
    size_t processed_frames = 0;

    for ( string &fpath : parse_list )
    {
        if ( !ros::ok() )
            break;

        cv::Mat input_image = cv::imread(fpath);
        if ( input_image.empty() )
            continue;

        cv::Mat source_image;
        input_image.copyTo(source_image);

        cv::blur(input_image, input_image, cv::Size(3, 3));

        std::vector<DetectionObject> corrected_dets;

        try {
            yolo.infer(input_image, corrected_dets);

            for (DetectionObject &det : corrected_dets)
            {
                string det_class_name = labels.at(det.cls_idx);

                /* Main color detection logic */
                if ( det_class_name== "traffic_light" )
                {
                    cv::Mat tl_frame = source_image(det.rect);

                    // Recognition required
                    cv::Mat hsv_frame;

                    cv::Mat green_frame;
                    cv::Mat red_frame;
                    cvtColor(tl_frame, hsv_frame, cv::COLOR_BGR2HSV);

                    cv::inRange( hsv_frame,
                                    cv::Scalar(77, 60, 236),
                                    cv::Scalar(94, 207, 255),
                                    green_frame );

                    cv::inRange( hsv_frame,
                                    cv::Scalar(0, 0, 222),
                                    cv::Scalar(80, 50, 255),
                                    red_frame );

                    int count_red = cv::countNonZero(red_frame);
                    int count_green = cv::countNonZero(green_frame);

                    if ( count_red > count_green )
                        det_class_name = "traffic_light_red";
                    else
                        det_class_name = "traffic_light_green";
                }

                cv::rectangle(input_image, det.rect, cv::Scalar(250, 0, 0), 2);
                cv::putText(input_image, det_class_name, det.rect.tl(),
                    cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, cv::Scalar(50,50,100), 1);
            }

            fs::path result_fpath = output_dir / fs::path(fpath).filename();

            cv::imwrite(result_fpath.string(), input_image);

        } catch (...) {

        }

        processed_frames++;
        if ( processed_frames % 10 == 0 )
        {
            cout << "Processed: " << processed_frames << " / " << total_frames << endl;
        }
    }

    return EXIT_SUCCESS;
}
