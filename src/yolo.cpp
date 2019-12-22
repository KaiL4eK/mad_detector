#include "yolo.hpp"

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
namespace pt = boost::property_tree;

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <chrono>
using namespace std;


double IntersectionOverUnion(const RawDetectionObject &box_1,
                             const RawDetectionObject &box_2)
{
    double width_of_overlap_area = fmin(box_1.x+box_1.w,
                                        box_2.x+box_2.w) -
                                   fmax(box_1.x,
                                        box_2.x);
    double height_of_overlap_area = fmin(box_1.y+box_1.h,
                                         box_2.y+box_2.h) -
                                    fmax(box_1.y,
                                         box_2.y);
    double area_of_overlap;
    if (width_of_overlap_area < 0 || height_of_overlap_area < 0)
        area_of_overlap = 0;
    else
        area_of_overlap = width_of_overlap_area * height_of_overlap_area;

    double box_1_area = box_1.h * box_1.w;
    double box_2_area = box_2.h * box_2.w;
    double area_of_union = box_1_area + box_2_area - area_of_overlap;

    return area_of_overlap / area_of_union;
}

CommonYOLO::CommonYOLO(std::string cfg_path) :
    mCfg(cfg_path)
{
}


YOLOConfig::YOLOConfig(string cfg_path)
{
    pt::ptree cfg_root;
    pt::read_json(cfg_path, cfg_root);

    pt::ptree model_root = cfg_root.get_child("model");

    /* Read anchors */
    cv::Point anchors_pair(-1, -1);

    _output_cnt = model_root.get_child("downsample").size();

    for (pt::ptree::value_type &v : model_root.get_child("labels"))
    {
        string label = v.second.get_value<string>();
        cout << label << endl;

        labels_.push_back(label);
    }

    for (pt::ptree::value_type &v : model_root.get_child("anchors"))
    {
        if (anchors_pair.x < 0)
        {
            anchors_pair.x = v.second.get_value<uint32_t>();
        }
        else
        {
            anchors_pair.y = v.second.get_value<uint32_t>();
            _anchors.push_back(anchors_pair);
            anchors_pair.x = -1; /* reset to read next number */
        }
    }

    vector<size_t> infer_size_raw;
    for (pt::ptree::value_type &v : model_root.get_child("infer_shape"))
    {
        infer_size_raw.push_back(v.second.get_value<uint32_t>());
    }

    for (pt::ptree::value_type &v : model_root.get_child("output_names"))
    {
        _output_names.push_back(v.second.get_value<string>());
    }

    for (pt::ptree::value_type &v : model_root.get_child("input_names"))
    {
        _input_names.push_back(v.second.get_value<string>());
    }

    if ( infer_size_raw.size() != 2 )
        throw invalid_argument("Invalid field 'model/infer_shape'");

    _infer_sz = cv::Size(infer_size_raw[1], infer_size_raw[0]);

    cout << "** Config **" << endl;
    cout << "Readed anchors: " << endl;
    for (cv::Point &pnt : _anchors)
    {
        cout << "  " << pnt << endl;
    }

    /* Read tile count */
    _tile_cnt = model_root.get_child("tiles").get_value<uint32_t>();

    cout << "Readed tiles count: " << _tile_cnt << endl;

    /* TODO - disable hardlink */
    _objectness_thresh = 0.5;
    _iou_threshold = 0.5;
}

std::vector<cv::Point> CommonYOLO::get_anchors(size_t layer_idx)
{
    vector<cv::Point> anchors;

    size_t anchors_per_output = mCfg._anchors.size() / mCfg._output_cnt;
    size_t start_idx = anchors_per_output * (mCfg._output_cnt - layer_idx - 1);
    size_t end_idx = anchors_per_output * (mCfg._output_cnt - layer_idx);

    cout << start_idx << " / " << end_idx << endl;

    for ( size_t i = start_idx; i < end_idx; i++ )
    {
        anchors.push_back(mCfg._anchors[i]);
    }

    return anchors;
}

std::vector<std::string> CommonYOLO::get_labels()
{
    vector<string> labels = mCfg.labels_;
    return labels;
}

void CommonYOLO::initResizeConfig(cv::Mat in_img,
                                  ImageResizeConfig &cfg)
{
    uint32_t new_w, new_h;

    cv::Size2f tile_sz;

    if (mCfg._tile_cnt == 1)
    {
        tile_sz = cv::Size2f(in_img.cols, in_img.rows);
        cfg.tile_rects.push_back( cv::Rect(cv::Point(0, 0), tile_sz) );
    }
    else if (mCfg._tile_cnt == 2)
    {
        tile_sz = cv::Size(in_img.cols/2, in_img.rows);
        cfg.tile_rects.push_back( cv::Rect(cv::Point(0, 0), tile_sz) );
        cfg.tile_rects.push_back( cv::Rect(cv::Point(in_img.cols/2, 0), tile_sz) );
    }

    if ( (mCfg._infer_sz.width / tile_sz.width) < (mCfg._infer_sz.height / tile_sz.height) )
    {
        new_w = mCfg._infer_sz.width;
        new_h = tile_sz.height / tile_sz.width * mCfg._infer_sz.width;
    }
    else
    {
        new_h = mCfg._infer_sz.height;
        new_w = tile_sz.width / tile_sz.height * mCfg._infer_sz.height;
    }

    cfg.top = (mCfg._infer_sz.height - new_h) / 2;
    cfg.bottom = (mCfg._infer_sz.height - new_h) - cfg.top;
    cfg.left = (mCfg._infer_sz.width - new_w) / 2;
    cfg.right = (mCfg._infer_sz.width - new_w) - cfg.left;

    cfg.new_sz = cv::Size(new_w, new_h);
    cfg.old_sz = in_img.size();

    cfg.offset = cv::Point2d(
        static_cast<float>(mCfg._infer_sz.width - cfg.new_sz.width) / 2. / mCfg._infer_sz.width,
        static_cast<float>(mCfg._infer_sz.height - cfg.new_sz.height) / 2. / mCfg._infer_sz.height
    );

    cfg.scale = cv::Point2d(
        static_cast<float>(cfg.new_sz.width) / mCfg._infer_sz.width,
        static_cast<float>(cfg.new_sz.height) / mCfg._infer_sz.height
    );
}

void CommonYOLO::resizeForNetwork(cv::Mat in_img,
                                  cv::Mat &out_img,
                                  ImageResizeConfig &cfg)
{
    cv::Mat tile_img = get_roi_tile(in_img, cfg.tile_idx);

    cv::resize(tile_img, tile_img, cfg.new_sz);

    cv::copyMakeBorder(tile_img, out_img,
                    cfg.top, cfg.bottom, cfg.left, cfg.right,
                    cv::BORDER_CONSTANT,
                    cv::Scalar(127, 127, 127));
}

void CommonYOLO::postprocessBoxes(std::vector<RawDetectionObject> &raw_boxes,
                                  ImageResizeConfig &cfg)
{
    /* For correction */
    for ( RawDetectionObject &det : raw_boxes )
    {
        // cout << "["
        //         << det.box_y-det.box_h/2 << ":"
        //         << det.box_y+det.box_h/2 << ", "
        //         << det.box_x-det.box_w/2 << ":"
        //         << det.box_x+det.box_w/2
        //         << "] "
        //         << det.conf << endl;

        if ( det.corrected )
            continue;

        float box_x = (det.x - cfg.offset.x) / cfg.scale.x * cfg.tile_rects[cfg.tile_idx].width;
        float box_y = (det.y - cfg.offset.y) / cfg.scale.y * cfg.tile_rects[cfg.tile_idx].height;
        det.w = det.w / cfg.scale.x * cfg.tile_rects[cfg.tile_idx].width;
        det.h = det.h / cfg.scale.y * cfg.tile_rects[cfg.tile_idx].height;
        det.x = box_x - det.w/2;
        det.y = box_y - det.h/2;

        det.x += cfg.tile_rects[cfg.tile_idx].x;
        det.y += cfg.tile_rects[cfg.tile_idx].y;

        det.xm = det.x + det.w;
        det.ym = det.y + det.h;

        det.corrected = true;
    }
}

void CommonYOLO::filterBoxes(std::vector<RawDetectionObject> &raw_boxes,
                             std::vector<DetectionObject> &result_boxes)
{
    std::sort(raw_boxes.begin(), raw_boxes.end(), std::greater<RawDetectionObject>());

    for (size_t i = 0; i < raw_boxes.size(); ++i)
    {
        RawDetectionObject &det = raw_boxes[i];

        if (det.conf == 0)
            continue;

        for (size_t j = i + 1; j < raw_boxes.size(); ++j)
        {
            if (/* det.cls_idx == raw_boxes[j].cls_idx && */
                IntersectionOverUnion(det, raw_boxes[j]) >= mCfg._iou_threshold)
            {
                raw_boxes[j].conf = 0;
            }
        }

        DetectionObject px_det;
        px_det.conf = det.conf;
        px_det.cls_idx = det.cls_idx;

        px_det.rect = cv::Rect(
            cv::Point( det.x, det.y ),
            cv::Point( det.xm, det.ym )
        );

        result_boxes.push_back(px_det);
    }
}

cv::Mat CommonYOLO::get_roi_tile(cv::Mat raw_image, size_t idx)
{
    if (mCfg._tile_cnt == 1)
    {
        return raw_image;
    }
    else if (mCfg._tile_cnt == 2)
    {
        if ( idx == 0 )
        {
            cv::Rect roi_left(cv::Point(0, 0), cv::Point(raw_image.cols/2, raw_image.rows));
            return raw_image(roi_left);
        }
        else if ( idx == 1 )
        {
            cv::Rect roi_right(cv::Point(raw_image.cols/2, 0), cv::Point(raw_image.cols, raw_image.rows));
            return raw_image(roi_right);
        }
    }
}

void CommonYOLO::get_detections(vector<RawDetectionObject> &dets, void *data, size_t grid_h, size_t grid_w, size_t chnls, vector<cv::Point> &anchors, ParsingFormat fmt)
{
    const size_t chnl_count = chnls;

    const float *detection = (float *)data;

    size_t c_stride;
    size_t h_stride;
    size_t w_stride;

    if ( fmt == ParsingFormat::HWC )
    {
        h_stride = grid_w * chnl_count;
        w_stride = chnl_count;
        c_stride = 1;
    }
    else if ( fmt == ParsingFormat::CHW )
    {
        c_stride = grid_h * grid_w;
        h_stride = grid_w;
        w_stride = 1;
    }

    const size_t class_count = chnl_count / anchors.size() - 5;
    const size_t box_count = class_count + 5;

    float obj_thresh = mCfg._objectness_thresh;

    for (size_t h_idx = 0; h_idx < grid_h; h_idx++)
    {
        for (size_t w_idx = 0; w_idx < grid_w; w_idx++)
        {
            size_t grid_offset = h_idx * h_stride + w_idx * w_stride;

            for (size_t anc_idx = 0; anc_idx < anchors.size(); anc_idx++)
            {
                RawDetectionObject det;
                size_t chnl_offset = anc_idx * box_count;

                // size_t box_idx_x = 0;
                // size_t box_idx_y = 1;
                // size_t box_idx_w = 2;
                // size_t box_idx_h = 3;
                // size_t obj_idx = 4;
                // size_t cls_idx = 5;

                float obj = detection[grid_offset + c_stride * (4 + chnl_offset)];
                obj = sigmoid(obj);
                if (obj < obj_thresh)
                    continue;

                det.x = detection[grid_offset + c_stride * (0 + chnl_offset)];
                det.y = detection[grid_offset + c_stride * (1 + chnl_offset)];
                det.w = detection[grid_offset + c_stride * (2 + chnl_offset)];
                det.h = detection[grid_offset + c_stride * (3 + chnl_offset)];

                det.w = anchors[anc_idx].x * exp(det.w) / mCfg._infer_sz.width;
                det.h = anchors[anc_idx].y * exp(det.h) / mCfg._infer_sz.height;
                det.x = (sigmoid(det.x) + w_idx) / grid_w;
                det.y = (sigmoid(det.y) + h_idx) / grid_h;

                for (size_t i_cls = 0; i_cls < class_count; i_cls++)
                {
                    float class_val = detection[grid_offset + c_stride * ((i_cls + 5) + chnl_offset)];
                    det.conf = sigmoid(class_val) * obj;

                    if ( det.conf < obj_thresh )
                        continue;

                    det.cls_idx = i_cls;
                    dets.push_back(det);
                }
            }
        }
    }
}
