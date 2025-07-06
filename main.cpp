#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

#define MAX_STRIDE 32
#define TARGET_SIZE 640

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

std::string getInputName(Ort::Session &session)
{
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::AllocatedStringPtr name_allocator = session.GetInputNameAllocated(0, allocator);
    return std::string(name_allocator.get());
}

std::string getOutputName(Ort::Session &session)
{
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::AllocatedStringPtr name_allocator = session.GetOutputNameAllocated(0, allocator);
    return std::string(name_allocator.get());
}

float clamp(float val, float min = 0.f, float max = 1280.f)
{
    return std::max(min, std::min(max, val));
}

void non_max_suppression(
    std::vector<Object> &proposals, std::vector<Object> &results,
    int orin_h, int orin_w, float conf_thres = 0.25f, float iou_thres = 0.65f)
{
    std::vector<cv::Rect> bboxes;
    std::vector<float> scores;
    std::vector<int> indices;

    for (const auto &obj : proposals)
    {
        bboxes.push_back(obj.rect);
        scores.push_back(obj.prob);
    }

    cv::dnn::NMSBoxes(bboxes, scores, conf_thres, iou_thres, indices);

    for (int i : indices)
    {
        results.push_back(proposals[i]);
    }
}

void preprocess(const cv::Mat &img, std::vector<float> &input_tensor, int target_size = TARGET_SIZE)
{
    cv::Mat resized, input;

    int img_w = img.cols;
    int img_h = img.rows;

    float scale = std::min(target_size / (float)img_w, target_size / (float)img_h);
    int new_w = img_w * scale;
    int new_h = img_h * scale;

    cv::resize(img, resized, cv::Size(new_w, new_h));

    int top = (target_size - new_h) / 2;
    int bottom = target_size - new_h - top;
    int left = (target_size - new_w) / 2;
    int right = target_size - new_w - left;

    cv::copyMakeBorder(resized, input, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    input.convertTo(input, CV_32F, 1.0 / 255);

    input_tensor.reserve(input.rows * input.cols * input.channels());
    for (int c = 0; c < input.channels(); ++c)
        for (int w = 0; w < input.cols; ++w)
            for (int h = 0; h < input.rows; ++h)
                input_tensor.push_back(input.at<cv::Vec3f>(h, w)[c]);
}

void postprocess(const float *output, int rows, int cols, float conf_threshold, std::vector<Object> &objects)
{
    for (int i = 0; i < rows; ++i)
    {
        float confidence = output[i * cols + 4]; // Object confidence

        if (confidence >= conf_threshold)
        {
            float x = output[i * cols];
            float y = output[i * cols + 1];
            float w = output[i * cols + 2];
            float h = output[i * cols + 3];
            int label = std::max_element(output + i * cols + 5, output + (i + 1) * cols) - (output + i * cols + 5);
            float prob = confidence;

            Object obj;
            obj.rect = cv::Rect_<float>(x - w / 2, y - h / 2, w, h);
            obj.label = label;
            obj.prob = prob;
            objects.push_back(obj);
        }
    }
}

void detect_yolov11(const cv::Mat &img, Ort::Session &session, int target_size, std::vector<Object> &objects)
{
    std::vector<float> input_tensor;

    std::cout << "target_size: " << target_size << std::endl;

    preprocess(img, input_tensor, target_size);

    std::vector<int64_t> input_shape = {1, 3, target_size, target_size};

    const char *input_name = getInputName(session).c_str();
    const char *output_name = getOutputName(session).c_str();

    Ort::MemoryInfo memory_info =
        Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator,
            OrtMemTypeDefault);
    Ort::Value input_tensor_ort =
        Ort::Value::CreateTensor<float>(
            memory_info,
            input_tensor.data(),
            input_tensor.size(),
            input_shape.data(),
            input_shape.size());

    auto output_tensors =
        session.Run(
            Ort::RunOptions{nullptr},
            &input_name,
            &input_tensor_ort,
            1,
            &output_name,
            1);

    float *output_data = output_tensors[0].GetTensorMutableData<float>();
    auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

    int rows = output_shape[1];
    int cols = output_shape[2];

    std::cout << "rows: " << rows << ", cols: " << cols << std::endl;

    postprocess(output_data, rows, cols, 0.5f, objects);
}

void draw_objects(const cv::Mat &img, const std::vector<Object> &objects)
{
    cv::Mat image = img.clone();

    for (const auto &obj : objects)
    {
        cv::rectangle(image, obj.rect, cv::Scalar(255, 0, 0), 2);

        std::string label = std::to_string(obj.label) + ": " + std::to_string(int(obj.prob * 100)) + "%";
        int baseline;
        cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);

        int x = std::max(0, std::min((int)obj.rect.x, image.cols - label_size.width));
        int y = std::max(label_size.height, (int)obj.rect.y);

        cv::rectangle(image, cv::Rect(cv::Point(x, y - label_size.height), label_size + cv::Size(0, baseline)),
                      cv::Scalar(255, 255, 255), cv::FILLED);
        cv::putText(image, label, cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }

    cv::imwrite("output.jpg", image);
}

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << " <model.onnx> <imagepath>" << std::endl;
        return -1;
    }

    const std::string model_path = argv[1];
    const std::string image_path = argv[2];
    int target_size = TARGET_SIZE;
    if (argc == 4)
    {
        const std::string side_size = argv[3];
        target_size = std::stoi(side_size);
    }

    cv::Mat img = cv::imread(image_path);
    if (img.empty())
    {
        std::cerr << "Failed to read image: " << image_path << std::endl;
        return -1;
    }

    Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR, "YOLOv11");
    Ort::SessionOptions session_options;
    Ort::Session session(env, model_path.c_str(), session_options);

    std::vector<Object> objects;
    detect_yolov11(img, session, target_size, objects);
    draw_objects(img, objects);

    return 0;
}
