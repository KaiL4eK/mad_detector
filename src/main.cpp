#include <boost/program_options.hpp>
namespace po = boost::program_options;

#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;

#include <boost/algorithm/string/predicate.hpp>
#include <boost/algorithm/string.hpp>

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
string g_input;
bool   g_compressed;
bool   g_debug_image_enabled;
int    g_target_rate;

string g_save_folder;

/********************************************************************/
/********************************************************************/

class FramesSource
{
public:
    virtual cv::Mat get_frame();

protected:
    cv::Mat frame_;

    mutex               frame_mutex_;
    condition_variable  frame_condvar_;
    bool is_empty_ = true;
};

cv::Mat FramesSource::get_frame()
{
    unique_lock<mutex> lock(frame_mutex_);

    while ( is_empty_ ) {
        frame_condvar_.wait(lock);
    }

    is_empty_ = true;
    return frame_;
}

/********************************************************************/
/********************************************************************/

class VideoFramesSource : public FramesSource
{
public:
    VideoFramesSource(string &filepath);

    cv::Mat get_frame();

private:
    cv::VideoCapture    cap_;
};

VideoFramesSource::VideoFramesSource(string &filepath)
{
    if ( !cap_.open(filepath) )
    {
        throw invalid_argument("Filepath is invalid (failed to open video): " + filepath);
    }
}

cv::Mat VideoFramesSource::get_frame()
{
    cv::Mat frame;
    cap_ >> frame;
    return frame;
}

/********************************************************************/
/********************************************************************/

class PictureFramesSource : public FramesSource
{
public:
    PictureFramesSource(string &filepath);

    cv::Mat get_frame();
};

PictureFramesSource::PictureFramesSource(std::string &filepath)
{
    frame_ = cv::imread(filepath);
}

cv::Mat PictureFramesSource::get_frame()
{
    cv::Mat new_frame;
    frame_.copyTo(new_frame);

    return new_frame;
}

/********************************************************************/
/********************************************************************/

class RosTopicFramesSource : public FramesSource
{
public:
    RosTopicFramesSource(ros::NodeHandle &nh, string &topic_name);

private:
    image_transport::ImageTransport it_;
    image_transport::Subscriber sub_;

    thread  polling_thread_;

    void polling_routine();
    void image_callback(const sensor_msgs::ImageConstPtr& msg);
    void image_compressed_callback(const sensor_msgs::CompressedImageConstPtr& msg);
};

RosTopicFramesSource::RosTopicFramesSource(ros::NodeHandle &nh, string &topic_name) :
    it_(nh)
{
    ROS_INFO_STREAM("Subscribing to " << topic_name);

    image_transport::TransportHints hints("compressed");

    if ( g_compressed )
        sub_ = it_.subscribe(topic_name, 1, &RosTopicFramesSource::image_callback, this, hints);
    else
        sub_ = it_.subscribe(topic_name, 1, &RosTopicFramesSource::image_callback, this);

    polling_thread_ = thread(&RosTopicFramesSource::polling_routine, this);
}

void RosTopicFramesSource::polling_routine()
{
    ROS_INFO_STREAM("Polling thread started");
    ros::spin();
}

void RosTopicFramesSource::image_callback(const sensor_msgs::ImageConstPtr& msg)
{
    unique_lock<mutex> lock(frame_mutex_);

    try
    {
        frame_ = cv_bridge::toCvShare(msg, "bgr8")->image;

        is_empty_ = false;
        frame_condvar_.notify_all();
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
    }
}

void RosTopicFramesSource::image_compressed_callback(const sensor_msgs::CompressedImageConstPtr& msg)
{
    unique_lock<mutex> lock(frame_mutex_);

    try
    {
        frame_ = cv::imdecode(cv::Mat(msg->data),1);

        is_empty_ = false;
        frame_condvar_.notify_all();
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("Could not convert to image!");
    }
}

/********************************************************************/
/********************************************************************/

#include <ctime>

class ImageSaver
{
public:
    ImageSaver(std::string &save_folder);

    void save_image(cv::Mat &frame, std::string filename);
    std::string get_datetime();

private:
    std::string dirpath_;
    size_t      indexer_;

};

ImageSaver::ImageSaver(std::string &save_folder) :
    indexer_(0)
{
    fs::path path = fs::path(save_folder) / fs::path(get_datetime());

    if ( !fs::exists(path) ) {
        fs::create_directory(path);
    }

    dirpath_ = path.string();
}

string ImageSaver::get_datetime()
{
    time_t rawtime;
    struct tm * timeinfo;
    char buffer[80];

    time(&rawtime);
    timeinfo = localtime(&rawtime);

    strftime(buffer,sizeof(buffer),"%d%m%Y_%H%M%S",timeinfo);
    return string(buffer);
}

void ImageSaver::save_image(cv::Mat &frame, std::string filename)
{
    fs::path fpath = fs::path(dirpath_) / fs::path(filename);
    string   fpath_str = fpath.string() + "_" + to_string(indexer_++) + ".png";

    ROS_DEBUG_STREAM_ONCE(fpath_str);

    cv::imwrite(fpath_str, frame);
}

/********************************************************************/
/********************************************************************/

class ROSImagePublisher
{
public:
    ROSImagePublisher(ros::NodeHandle &nh, std::string topic_name);
    void send_rgb(cv::Mat &frame);

private:
    image_transport::Publisher      pub_;
    image_transport::ImageTransport it_;
};


ROSImagePublisher::ROSImagePublisher(ros::NodeHandle &nh, std::string topic_name) :
    it_(nh)
{
    pub_ = it_.advertise(topic_name, 1);
}


void ROSImagePublisher::send_rgb(cv::Mat &frame)
{
    sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", frame).toImageMsg();
    pub_.publish(msg);
}

/********************************************************************/
/********************************************************************/

int main(int argc, char **argv)
{
    ros::init(argc, argv, "yolo_detector");
    ros::NodeHandle pr_nh("~");
    ros::NodeHandle nh;

    pr_nh.getParam("config_path", g_cfg_path);
    pr_nh.getParam("ir_path", g_ir_path);
    pr_nh.getParam("input", g_input);
    pr_nh.param<int>("processing_rate", g_target_rate, 0);
    pr_nh.param<bool>("compressed", g_compressed, false);
    pr_nh.param<bool>("debug_image_enabled", g_debug_image_enabled, false);
    pr_nh.param<string>("device", g_device_type, "CPU");
    pr_nh.param<string>("save_folder", g_save_folder, "");

    shared_ptr<FramesSource> source;
    /* TODO - check working with Upper case in topics */
    boost::algorithm::to_lower(g_input);

    /* As device - same like video */
    if ( boost::starts_with(g_input, "/dev") ) {
        source = make_shared<VideoFramesSource>(g_input);
    /* As video */
    } else if ( boost::ends_with(g_input, ".mp4") ) {
        source = make_shared<VideoFramesSource>(g_input);
    /* As picture */
    } else if ( boost::ends_with(g_input, ".png") ) {
        source = make_shared<PictureFramesSource>(g_input);
    } else if ( boost::ends_with(g_input, ".jpg") ) {
        source = make_shared<PictureFramesSource>(g_input);
    /* As topic, at last =) */
    } else {
        source = make_shared<RosTopicFramesSource>(nh, g_input);
    }

    ros::Publisher pub = nh.advertise<mad_detector::Detections>("detections", 1000);

    ROSImagePublisher debug_image_pub(nh, "debug_detections_image");

    shared_ptr<ImageSaver> image_saver;
    if ( !g_save_folder.empty() ) {
        image_saver = make_shared<ImageSaver>(g_save_folder);
    }

    YOLO_OpenVINO yolo(g_cfg_path);
    yolo.init(g_ir_path, g_device_type);

    vector<string> labels = yolo.get_labels();

    if ( g_target_rate <= 0 )
        g_target_rate = 30;

    ros::Rate loop_rate(g_target_rate);

    while ( ros::ok() )
    {
        mad_detector::Detections   output_dets;

        cv::Mat input_image = source->get_frame();
        if ( input_image.empty() )
            break;

        cv::Mat source_image;
        input_image.copyTo(source_image);

        cv::blur(input_image, input_image, cv::Size(3, 3));

        std::vector<DetectionObject> corrected_dets;

        try {
            yolo.infer(input_image, corrected_dets);

            for (DetectionObject &det : corrected_dets)
            {
                // cout << "Detection: " << labels.at(det.cls_idx) << endl;

                mad_detector::Detection topic_detection;
                topic_detection.object_class = labels.at(det.cls_idx);
                topic_detection.probability = det.conf;

                // cout << det.rect << endl;

                /* Main color detection logic */
                if ( labels.at(det.cls_idx) == "traffic_light" )
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
                        topic_detection.object_class = "traffic_light_red";
                    else
                        topic_detection.object_class = "traffic_light_green";
                }

                if ( g_debug_image_enabled )
                {
                    cv::rectangle(input_image, det.rect, cv::Scalar(250, 0, 0), 2);
                    cv::putText(input_image, topic_detection.object_class, det.rect.tl(),
                        cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, cv::Scalar(50,50,100), 1);
                }

                /* ROS message preparation */
                geometry_msgs::Point size_px;
                size_px.x = det.rect.width;
                size_px.y = det.rect.height;

                geometry_msgs::Point ul_point;
                ul_point.x = det.rect.tl().x * 1. / source_image.cols;
                ul_point.y = det.rect.tl().y * 1. / source_image.rows;

                geometry_msgs::Point br_point;
                br_point.x = det.rect.br().x * 1. / source_image.cols;
                br_point.y = det.rect.br().y * 1. / source_image.rows;

                topic_detection.ul_point = ul_point;
                topic_detection.br_point = br_point;
                topic_detection.size_px = size_px;

                output_dets.detections.push_back(topic_detection);
            }

            if ( !output_dets.detections.empty() )
                pub.publish(output_dets);

            if ( g_debug_image_enabled ) {
                debug_image_pub.send_rgb(input_image);
            }

            if ( image_saver ) {
                image_saver->save_image(source_image, "src");
                image_saver->save_image(input_image, "det");
            }
        } catch (...) {

        }

        loop_rate.sleep();
    }

    return EXIT_SUCCESS;
}
