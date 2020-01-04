#include <boost/program_options.hpp>
namespace po = boost::program_options;

#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;

#include <boost/algorithm/string/predicate.hpp>

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
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

string g_ir_path;
string g_cfg_path;
string g_device_type;
string g_input;
bool   g_compressed;

string g_save_folder;

// ostream &operator<<(ostream &out, const vector<size_t> &c)
// {
//     out << "[";
//     for (const size_t &s : c)
//         out << s << ",";
//     out << "]";
//     return out;
// }

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

class VideoFramesSource : public FramesSource
{

};

class PictureFramesSource : public FramesSource
{

};

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

class ImageSaver
{
public:
    ImageSaver(std::string &save_folder);

    void save_image(cv::Mat &frame, std::string &filename);

private:
    std::string &save_folder_;
    size_t      indexer_;

};

ImageSaver::ImageSaver(std::string &save_folder) :
    save_folder_(save_folder),
    indexer_(0)
{
    if ( !fs::exists(save_folder_) ) {
        fs::create_directory(save_folder_);
    }
}

void ImageSaver::save_image(cv::Mat &frame, std::string &filename)
{
    fs::path fpath = fs::path(save_folder_) / fs::path(filename);
    string   fpath_str = fpath.string() + "_" + to_string(indexer_++) + ".png";

    ROS_DEBUG_STREAM_ONCE(fpath_str);

    cv::imwrite(fpath_str, frame);
}


int main(int argc, char **argv)
{
    ros::init(argc, argv, "yolo_detector");
    ros::NodeHandle pr_nh("~");
    ros::NodeHandle nh;

    pr_nh.getParam("config_path", g_cfg_path);
    pr_nh.getParam("ir_path", g_ir_path);
    pr_nh.getParam("device", g_device_type);
    pr_nh.getParam("input", g_input);
    pr_nh.getParam("compressed", g_compressed);
    pr_nh.getParam("save_folder", g_save_folder);

    shared_ptr<FramesSource> source;

    if ( boost::starts_with(g_input, "/dev") ) {

    } else if ( boost::ends_with(g_input, ".mp4") ) {

    } else if ( boost::ends_with(g_input, ".png") ) {

    } else if ( boost::ends_with(g_input, ".jpg") ) {

    } else {
        source = make_shared<RosTopicFramesSource>(nh, g_input);
    }

    shared_ptr<ImageSaver> image_saver;
    if ( !g_save_folder.empty() ) {
        image_saver = make_shared<ImageSaver>(g_save_folder);
    }

    YOLO_OpenVINO yolo(g_cfg_path);
    yolo.init(g_ir_path, g_device_type);

    vector<string> labels = yolo.get_labels();

    while ( ros::ok() )
    {
        cv::Mat input_image = source->get_frame();
        if ( input_image.empty() )
            break;

        cv::Mat source_image;
        input_image.copyTo(source_image);

        cv::blur(input_image, input_image, cv::Size(3, 3));

        std::vector<DetectionObject> corrected_dets;
        yolo.infer(input_image, corrected_dets);

        for (DetectionObject &det : corrected_dets)
        {
            cv::rectangle(input_image, det.rect, cv::Scalar(250, 0, 0), 2);

            cv::putText(input_image, labels.at(det.cls_idx), det.rect.tl(),
                cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, cv::Scalar(50,50,100), 1, CV_AA);

            // cout << "Detection: " << labels.at(det.cls_idx) << endl;
        }

        cv::Mat resized;
        cv::resize(input_image, resized, cv::Size(800, 600));

        cv::imshow("Boxes", resized);
        if ( cv::waitKey(1) == 27 )
            break;

        if ( image_saver && corrected_dets.size() > 0 ) {
            std::string image_prefix = "saved";
            image_saver->save_image(source_image, image_prefix);
        }
    }

    return EXIT_SUCCESS;
}
