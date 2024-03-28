#include <iostream>
#include <opencv2/opencv.hpp>
#include <filesystem>

namespace fs = std::filesystem;

// Save Image in directory
void save_image(const std::string& class_label, int count, const cv::Mat& roi, const std::string& directory) {
    cv::imwrite(directory + "/" + class_label + "/" + std::to_string(count) + ".jpg", roi);
}

int main() {
    std::string directory = "dataSet/";
    int minValue = 70;
    if (!fs::exists(directory))
        fs::create_directories(directory);
    std::vector<std::string> class_labels = { "zero", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z" };
    for (const auto& label : class_labels) {
        std::string class_directory = directory + "/" + label;
        if (!fs::exists(class_directory))
            fs::create_directories(class_directory);
    }
    std::map<std::string, int> class_counts;
    for (const auto& label : class_labels)
        class_counts[label] = fs::directory_iterator(directory + "/" + label) == fs::directory_iterator() ? 0 : std::distance(fs::directory_iterator(directory + "/" + label), fs::directory_iterator());
    std::map<int, std::string> key_map;
    for (const auto& label : class_labels)
        key_map[label[0]] = label;

    cv::VideoCapture cap(0);
    int interrupt = -1;
    while (true) {
        cv::Mat frame;
        cap.read(frame);
        cv::flip(frame, frame, 1);

        for (const auto& [label, count] : class_counts) {
            std::string text = label + ": " + std::to_string(count);
            cv::putText(frame, text, cv::Point(10, 60 + 20 * (label[0] - 'a')), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 255), 1);
        }

        int x1 = 0.5 * frame.cols, y1 = 10, x2 = frame.cols - 10, y2 = 0.5 * frame.rows;
        cv::rectangle(frame, cv::Point(x1 - 1, y1 - 1), cv::Point(x2 + 1, y2 + 1), cv::Scalar(255, 0, 0), 1);
        cv::Mat roi = frame(cv::Rect(x1, y1, x2 - x1, y2 - y1));
        cv::Mat test_image = roi.clone();
        cv::resize(test_image, test_image, cv::Size(300, 300));
        cv::imshow("test", test_image);
        cv::imshow("Frame", frame);
        interrupt = cv::waitKey(10);
        if ((interrupt & 0xFF) == 27) // esc key
            break;
        else if (key_map.find(interrupt & 0xFF) != key_map.end()) {
            std::string class_label = key_map[interrupt & 0xFF];
            save_image(class_label, class_counts[class_label], test_image, directory);
            class_counts[class_label]++;
        }
    }
    cap.release();
    cv::destroyAllWindows();
    return 0;
}
