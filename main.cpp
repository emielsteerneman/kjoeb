#include <iostream>

#include <opencv2/opencv.hpp>
#include <vector>
#include <tuple>
#include <numeric>
#include <limits>

const int CUBE_SIZE = 3;

// Generate a checkered pattern for performance testing
void g(){

    int nSquares = 20;
    int nSize = 1000;

    cv::Mat frame = cv::Mat::zeros(nSize, nSize, CV_8UC3);

    int squareSize = nSize / nSquares;
    cv::Point br(squareSize - 20, squareSize - 20);

    for(int x = 0;  x < nSquares; x++){
        for(int y = 0; y < nSquares; y++){
            cv::Point p1(x * squareSize + 10, y * squareSize + 10);
            cv::rectangle(frame, p1, p1 + br, {255, 255, 255}, -1);
        }
    }
    cv::imshow("checkers", frame);
    cv::imwrite("../checkers.png", frame);
    cv::waitKey(0);
}

// ====================================================================================
// ================================ IMPRESSIVE LOGGING ================================
// ====================================================================================

class Timer {
    std::chrono::steady_clock::time_point t_begin;
    std::chrono::steady_clock::time_point t_end;
public:
    Timer(){};
    void start(){ t_begin = std::chrono::steady_clock::now(); }
    void stop() { t_end   = std::chrono::steady_clock::now(); }
    double get(){ return std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_begin).count() / 1000.0; }
};

class VidWriter{
    int width, height, nWidth, nHeight;
    int subWidth, subHeight, at;
    int fps;
    cv::VideoWriter writer;
    cv::Mat frame, workFrame;
    bool disabled = false;
public:
    VidWriter(std::string filename, int w, int h, int nW, int nH, int fps)
        : width(w), height(h), nWidth(nW), nHeight(nH), fps(fps) {
        subWidth = width / nWidth;
        subHeight = height / nHeight;
        bool writerOpened = writer.open(filename, cv::VideoWriter::fourcc('a', 'v', 'c', '1'), fps, {w, h}, true);
        if(!writerOpened)
            std::cout << "[VidWriter] Warning! Could not open stream!" << std::endl;
        frame = cv::Mat::zeros(height, width, CV_8UC3);
        reset();
    }
    void disable(){disabled = true;}
    void reset(){ /*frame = cv::Mat::zeros(height, width, CV_8UC3);*/ at = 0;}
    void add(const cv::Mat& subFrame, std::string text = ""){
        if(disabled) return;
        Timer t;
        t.start();

        // Resize
        cv::resize(subFrame, workFrame, {width / nWidth, height / nHeight});
//
//        std::cout << " Adding frame " << at << " | " << text << std::endl;
        if(!writer.isOpened())
            std::cout << "[VidWriter] Warning! Stream not opened" << std::endl;
        if(9 <= at)
            std::cout << "[VidWriter] Warning! at=" << at << std::endl;

        int toWidth = (at % 3) * (width/nWidth);
        int toHeight = (int)floor(at/3) * (height/nHeight);
        cv::Rect dstRect(toWidth, toHeight, subWidth, subHeight);
        workFrame.copyTo(frame(dstRect));

        if(text != ""){
            cv::rectangle(frame, {toWidth, toHeight, width, 24}, {0, 0, 0}, -1);
            cv::putText(frame, text, {toWidth+5, toHeight+16}, cv::FONT_HERSHEY_SIMPLEX, 0.5, {255, 255, 255}, 1, cv::LINE_AA);
        }

        cv::line(frame, {toWidth, toHeight}, {toWidth + width, toHeight}, {100, 100, 100});
        cv::line(frame, {toWidth, toHeight}, {toWidth, toHeight + height}, {100, 100, 100});

        at++;
        t.stop();
//        std::cout << "[VidWriter] writing took " << t.get() << "ms" << std::endl;
    }

    void show(){
        if(disabled) return;
        cv::imshow("VideoWriter", frame);
        cv::namedWindow("VideoWriter", cv::WND_PROP_FULLSCREEN);
        cv::setWindowProperty("VideoWriter", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
//        reset();
    }

    void flush(){
        if(disabled) return;
        writer.write(frame);
        reset();
    }

};

struct ExtendedRect : cv::Rect {
    double angle;
    double sizeNorm;
    ExtendedRect(const cv::Rect& boundingBox, double sizeNorm = 0.0, double angle = 0.0)
    : cv::Rect(boundingBox), sizeNorm(sizeNorm), angle(angle){};
};

using Line = std::tuple<int, const cv::Rect&, const cv::Rect&, double>;



template<class T> using Cluster = std::vector<T>;
template<class T> using SuperCluster = std::vector<std::vector<T>>;

// TODO What if either a of b equals 0
bool inRangeRel(double a, double b, double c = 0.85){
    return c < a/b && c < b/a;
}

bool inRangeAbs(double a, double b, double c){
    return fabs(a-b) <= c;
}

int findRects(cv::Mat& img, std::vector<cv::Rect>& rects){
    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;

    findContours(img, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

    for(const auto& contour : contours) {
        cv::Rect box = cv::boundingRect(contour);
        if(500 < box.area())
            if(inRangeRel(box.width, box.height))
                rects.push_back(box);
    }
    return contours.size();
}


// ====================================================================================
// ==================================== EPIC MATHS ====================================
// ====================================================================================

double average(const std::vector<int>& vec){
    int sum = 0;
    for(int i = 0; i < vec.size(); i++)
        sum += vec[i];
    return sum / (double)vec.size();
}

double variance(const std::vector<int>& vec){
    double avg = average(vec);
    double sum = 0.0;
    for(int i = 0; i < vec.size(); i++)
        sum += std::pow(vec[i] - avg, 2);
    return sum / (double)(vec.size() - 2);

}

//http://supp.iar.com/FilesPublic/SUPPORT/000419/AN-G-002.pdf
unsigned int root(unsigned int x){
    unsigned int a,b;
    b     = x;
    a = x = 0x3f;
    x     = b/x;
    a = x = (x+a)>>1;
    x     = b/x;
    a = x = (x+a)>>1;
    x     = b/x;
    x     = (x+a)>>1;
    return(x);
}

double normalizeAngle90(double angle){
    angle *= 57.29;
    double step = 90;

    angle = fabs(angle);
    while(step <= angle)
        angle -= step;

    if(step / 2 < angle)
        angle -= step;

    return angle / 57.29;
}

// Distance calculation without root
int calcDistance(const cv::Rect &r1, const cv::Rect &r2){
    int dx = r1.x - r2.x;
    int dy = r1.y - r2.y;
    return dx * dx + dy * dy;
}

double distanceCorrect(const cv::Rect& r1, const cv::Rect& r2){
    return sqrt(calcDistance(r1, r2));
}

bool calcRectsIntersect(const cv::Rect& r1, const cv::Rect& r2){
    return r1.x < (r2.x + r2.width)
        && r2.x < (r1.x + r1.width)
        && r1.y < (r2.y + r2.height)
        && r2.y < (r1.y + r1.height);
}

bool calcRectsIntersect(const cv::Point& a11, const cv::Point& a12, const cv::Point& b11, const cv::Point& b12){
    return calcRectsIntersect({a11, a12}, {b11, b12});
}

bool calcRectsIntersect(const cv::Rect& r11, const cv::Rect& r12, const cv::Rect& r21, const cv::Rect& r22){
    return calcRectsIntersect(r11.tl(), r12.tl(), r21.tl(), r22.tl());
}

bool calcLinesIntersect(const Line& l1, const Line& l2){
    return calcRectsIntersect(std::get<1>(l1), std::get<2>(l1), std::get<1>(l2), std::get<2>(l2));
}

cv::Point rotatePointAroundPoint(cv::Point p, cv::Point center, double angle){
    double s = std::sin(angle);
    double c = std::cos(angle);
    p -= center;    // Translate to origin
    int xNew = (int)(p.x * c - p.y * s);
    int yNew = (int)(p.x * s + p.y * c);

    return cv::Point(xNew, yNew) + center;
}

cv::Rect getBoundingBox(const Cluster<cv::Rect> rects){
    cv::Rect boundingBox;
    int xMin = 999999;
    int xMax = 0;
    int yMin = 999999;
    int yMax = 0;
    for(const ExtendedRect& rect : rects){
        xMin = std::min(xMin, rect.x);
        xMax = std::max(xMax, rect.x + rect.width);
        yMin = std::min(yMin, rect.y);
        yMax = std::max(yMax, rect.y + rect.height);
    }
    boundingBox.x = xMin;
    boundingBox.width = xMax - xMin;
    boundingBox.y = yMin;
    boundingBox.height = yMax - yMin;
    return boundingBox;
}

cv::Rect getBoundingBox(const Cluster<ExtendedRect> rects){
    cv::Rect boundingBox;
    int xMin = 999999;
    int xMax = 0;
    int yMin = 999999;
    int yMax = 0;
    for(const ExtendedRect& rect : rects){
        xMin = std::min(xMin, rect.x);
        xMax = std::max(xMax, rect.x + rect.width);
        yMin = std::min(yMin, rect.y);
        yMax = std::max(yMax, rect.y + rect.height);
    }
    boundingBox.x = xMin;
    boundingBox.width = xMax - xMin;
    boundingBox.y = yMin;
    boundingBox.height = yMax - yMin;
    return boundingBox;
}

// ====================================================================================
// ================================ INSANE  CLUSTERING ================================
// ====================================================================================

template <class T>
bool sortClustersBySize(const std::vector<T> &c1, const std::vector<T> &c2){
    return c1.size() < c2.size();
}

// Find the cluster with the best lines-to-intersections ratio (lines/intersections)
// This can be done backward, and stopping when cluster.size() < bestRatio
int selectLinesByIntersectRatio(const SuperCluster<Line> lineClusters){
    Timer t;
    t.start();

    int iBestCluster = 0;
    double bestRatio = 0;

    // For each cluster, count the intersections based on bounding boxes of lines
    for(int iCluster = lineClusters.size()-1; 0 <= iCluster; iCluster--){
        const auto& cluster = lineClusters[iCluster];
        if(cluster.size() < bestRatio)
            break;

        int iIntersections = 0;
        // Compare each line with eachother, increment intersections if needed
        for (int iLine = 0; iLine < cluster.size(); iLine++)
            for (int jLine = iLine + 1; jLine < cluster.size(); jLine++)
                if(calcLinesIntersect(cluster[iLine], cluster[jLine]))
                    iIntersections++;
        // +0.5 to compensate for cluster/0 vs cluster/1
        double ratio = iIntersections == 0 ? cluster.size() + 0.5 : cluster.size() / (float)iIntersections;
//                std::cout << cluster.size() << " / " << iIntersections << " = " << ratio << std::endl;
        if(bestRatio < ratio){
            bestRatio = ratio;
            iBestCluster = iCluster;
        }
    }

    t.stop();
    std::cout << "[clustered lines]        time=" << t.get() << "ms bestCluster=" << iBestCluster << " size =" << lineClusters[iBestCluster].size() << " ratio=" << bestRatio << std::endl;
    return iBestCluster;
}

// TODO don't cluster based on any match. Cluster based on median / mean / variance / etc.
void clusterRectsByArea(const std::vector<cv::Rect> &rects, SuperCluster<cv::Rect> &clusters){
    Timer t;
    t.start();

    for(const cv::Rect& rect : rects){
        bool found = false;
        for(auto& cluster : clusters){
            for(const auto& rectComp : cluster){
                if(found) break;
                if(inRangeRel(rect.area(), rectComp.area(), 0.90)){    // Should be compared with average / median instead of all.
                    found = true;
                    cluster.push_back(rect);
                }
            }
        }

        if(!found){
            clusters.emplace_back();
            clusters.back().push_back(rect);
        }
    }
    std::sort(clusters.begin(), clusters.end(), sortClustersBySize<cv::Rect>);

    t.stop();
    std::cout << "[clusterRectsByArea]     time=" << t.get() << "ms rects=" << rects.size() << " clusters=" << clusters.size() << std::endl;
}

void clusterRectsByDistance(const Cluster<cv::Rect> &rects, SuperCluster<Line> &clusters, double threshold = 0.9, int maxDistance = std::numeric_limits<int>::max(), int minDistance = 0){
    Timer t;
    t.start();

    // Find all lines between all rects
    int max = maxDistance * maxDistance;    // because not using root
    int min = minDistance * minDistance;    // because not using root
    std::vector<Line> lines;
    for(int iRect = 0; iRect < rects.size(); iRect++) {
        for (int jRect = iRect + 1; jRect < rects.size(); jRect++) {
            int distance = calcDistance(rects[iRect], rects[jRect]);
            if(min <= distance && distance <= max) {
                double angle = std::atan2(rects[iRect].y - rects[jRect].y, rects[iRect].x - rects[jRect].x);
                lines.emplace_back(std::forward_as_tuple(
                        calcDistance(rects[iRect], rects[jRect]),
                        rects[iRect],
                        rects[jRect],
                        angle
                ));
            }
        }
    }

    // Cluster lines by length
    for(const Line& line : lines){
        bool found = false;
        for(auto& cluster : clusters){
            for(const auto& lineComp : cluster){
                if(found) break;
                if(inRangeRel(std::get<0>(line), std::get<0>(lineComp), threshold)){    // Should be compared with average / median instead of all.
                    found = true;
                    cluster.push_back(line);
                }
            }
        }

        if(!found){
            clusters.emplace_back();
            clusters.back().push_back(line);
        }
    }
    std::sort(clusters.begin(), clusters.end(), sortClustersBySize<Line>);

    t.stop();
    std::cout << "[clusterRectsByDistance] time=" << t.get() << "ms lines=" << lines.size() << " clusters=" << clusters.size() << std::endl;
}

void clusterLinesByAngle(const Cluster<Line>& lines, SuperCluster<Line> &clusters){
    Timer t;
    t.start();

    double pi_step = M_PI / 4;
    double pi_threshold = pi_step * 0.10;

    for(const Line& line : lines){
        bool found = false;
        for(auto& cluster : clusters){
            for(const Line&  lineComp: cluster){
                if(found) break;

                const cv::Rect& l1r1 = std::get<1>(line);
                const cv::Rect& l1r2 = std::get<2>(line);
                const cv::Rect& l2r1 = std::get<1>(lineComp);
                const cv::Rect& l2r2 = std::get<2>(lineComp);

                double a1 = std::atan2(l1r1.y - l1r2.y, l1r1.x - l1r2.x);
                double a2 = std::atan2(l2r1.y - l2r2.y, l2r1.x - l2r2.x);

                double angle = fabs(a1-a2);
                bool isInRange = inRangeAbs(angle, 0, pi_threshold)
                        || inRangeAbs(angle, pi_step * 1, pi_threshold)
                        || inRangeAbs(angle, pi_step * 2, pi_threshold)
                        || inRangeAbs(angle, pi_step * 3, pi_threshold)
                        || inRangeAbs(angle, pi_step * 4, pi_threshold);

//                std::cout << a1 << " | " << a2 << " | " << angle << " | " << isInRange << std::endl;


                if(isInRange){
                    found = true;
                    cluster.push_back(line);
                }
            }
        }

        if(!found){
            clusters.emplace_back();
            clusters.back().push_back(line);
        }
    }
    std::sort(clusters.begin(), clusters.end(), sortClustersBySize<Line>);

    t.stop();
    std::cout << "[clusterLinesByAngle]    time=" << t.get() << "ms lines=" << lines.size() << " clusters=" << clusters.size() << std::endl;
}

// Rects are duplicate if the distance between their centers are small compared to their sizes
void deduplicateRects(const std::vector<cv::Rect> &in, std::vector<cv::Rect>& out){
    Timer t;
    t.start();

    for(const cv::Rect& rect : in){
        bool found = false;
        for(const cv::Rect& rectComp : out){
            double dist = calcDistance(rect, rectComp);
            if(dist < 0.75 * rect.area()) {
                found = true;
                break;
            }
        }
        if(!found)
            out.push_back(rect);
    }

    t.stop();
    std::cout << "[deduplicateRects]       time=" << t.get() << "ms " << in.size() << " -> " << out.size() << std::endl;
}



// ====================================================================================
// ================================ SWEET SOFT DRAWING ================================
// ====================================================================================

void DrawRotatedRectangle(cv::Mat& image, const ExtendedRect& rect){
    cv::Scalar color = cv::Scalar(255.0, 255.0, 255.0); // white

    // Create the rotated rectangle
    cv::Point centerPoint((int)(rect.x + rect.sizeNorm / 2), (int)(rect.y + rect.sizeNorm / 2));
    cv::RotatedRect rotatedRectangle(centerPoint, {(float)rect.sizeNorm, (float)rect.sizeNorm}, (int)(rect.angle * 57.2958));

    // We take the edges that OpenCV calculated for us
    cv::Point2f vertices2f[4];
    rotatedRectangle.points(vertices2f);

    // Convert them so we can use them in a fillConvexPoly
    cv::Point vertices[4];
    for(int i = 0; i < 4; ++i){
        vertices[i] = vertices2f[i];
    }

    for(int i = 0; i < 4; i++){
        cv::line(image, vertices[i], vertices[(i+1)%4], color, 2);
    }

    // Now we can fill the rotated rectangle with our specified color
//    cv::fillConvexPoly(image,vertices,4,color);
}

int main() {
    std::cout << "Hello, World!" << std::endl;
    std::cout << std::setprecision(2) << std::fixed;

    const int FPS = 10;

    Timer t;

    cv::VideoCapture cap;
    if(!cap.open(0)) {
        std::cout << "Could not open capture device" << std::endl;
        return 0;
    }

    cap.grab();

    // Set some constants
    const int WIDTH  = 640;//(int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    const int HEIGHT = 360;//(int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);

    // 1920x1080 -> 5000 < area
    // 1280x720  -> 2000 < area

    const int LINE_MAX_LENGTH = std::min(WIDTH, HEIGHT) / 3;
    const int RECT_MAX_AREA   = LINE_MAX_LENGTH * LINE_MAX_LENGTH;


    std::cout << "CAP_PROP_FRAME_WIDTH  " << cap.get(cv::CAP_PROP_FRAME_WIDTH) << std::endl;
    std::cout << "CAP_PROP_FRAME_HEIGHT " << cap.get(cv::CAP_PROP_FRAME_HEIGHT) << std::endl;
    std::cout << "CAP_PROP_FPS          " << cap.get(cv::CAP_PROP_FPS) << std::endl;

    if(!cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G')))
        std::cout << "Could not set CAP_PROP_FOURCC" << std::endl;

    if(!cap.set(cv::CAP_PROP_FRAME_WIDTH, WIDTH))
        std::cout << "Could not set CAP_PROP_FRAME_WIDTH" << std::endl;

    if(!cap.set(cv::CAP_PROP_FRAME_HEIGHT, HEIGHT))
        std::cout << "Could not set CAP_PROP_FRAME_HEIGHT" << std::endl;

//    if(!cap.set(cv::CAP_PROP_FPS, 30))
//        std::cout << "Could not set CAP_PROP_FPS" << std::endl;

    cv::Mat frame, frame_og, workFrame, workFrameClustering, mask, mask2;
    cv::Mat channels[3];

    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;
    std::vector<cv::Rect> rectangles;


    int erosion_size = 1;
    cv::Mat element = getStructuringElement(cv::MORPH_CROSS,
            cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1),
            cv::Point(erosion_size, erosion_size) );

    VidWriter writer("/home/emiel/Desktop/_kjoeb.mp4", 1280, 720, 3, 3, FPS);
//    writer.disable();

    int nFrames = 0;

    for(;;){
        std::cout << nFrames << std::endl;
        nFrames++;

        writer.show();
        writer.reset();
//        writer.flush();

//        if(1000 < nFrames)
//            break;

        // Grab frame
        if(cv::waitKey(10) == 27 ) break; // esc
//        cv::destroyAllWindows();

        // Clear the annoying camera buffer -> always get the latest frame
//        for(int i = 0; i < 5; i++)
//            cap.grab();

        t.start();
        bool isCaptured = cap.read(frame_og);
        frame_og.copyTo(frame);
        t.stop();
        std::cout << "[capture]            time=" << t.get() << "ms" << std::endl;

        if(frame.empty()){
            std::cout << "Frame is empty!" << std::endl;
            continue;
        }

        // Shrink frame
//        cv::resize(frame, frame, {frame.cols * 1 / 4, frame.rows * 1 / 4});
//        cv::flip(frame, frame, 1);
        writer.add(frame, "Original Frame " + std::to_string(nFrames));

        frame.copyTo(workFrame);
        cv::blur(workFrame , workFrame, {5, 5});

        // Laplacian edge detection
        t.start();
        cv::Laplacian(workFrame, workFrame, CV_8UC3, 3);
        cv::normalize(workFrame, workFrame, 0, 255, cv::NORM_MINMAX, CV_32FC1);
        cv::convertScaleAbs(workFrame, workFrame);
//        workFrame *= 10;

        // Create mask and apply
        cv::inRange(workFrame, cv::Scalar(0, 0, 0), cv::Scalar(20, 20, 20), mask);
        mask = 255 - mask;
//        cv::erode(mask, mask, element);
        cv::dilate(mask, mask, element, cv::Point(), 3);

        cv::Mat workFrameMasked;
        workFrame.copyTo(workFrameMasked, mask);
//        workFrame.copyTo(workFrameMasked);
        t.stop();

        writer.add(workFrameMasked, "Laplacian + mask " + std::to_string(t.get()));
        std::cout << "[laplacian]              time=" << t.get() << "ms" << std::endl;

        cv::cvtColor(mask, mask, cv::COLOR_GRAY2BGR);
        writer.add(mask, "inRange");

        int nBoxes = 0;
        int nRectangles = 0;

        cv::split(workFrameMasked, channels);
        if(true) {
//            workFrameClustering = cv::Mat::zeros(frame.rows, frame.cols, CV_8UC3);
            rectangles.clear();
            nBoxes += findRects(channels[0], rectangles);
            nRectangles += rectangles.size();
            for(cv::Rect rect : rectangles)
                cv::rectangle(workFrameMasked, rect, {255, 0, 0});

            rectangles.clear();
            nBoxes += findRects(channels[1], rectangles);
            nRectangles += rectangles.size();
            for(cv::Rect rect : rectangles)
                cv::rectangle(workFrameMasked, rect, {0, 255, 0});

            rectangles.clear();
            nBoxes += findRects(channels[2], rectangles);
            nRectangles += rectangles.size();
            for(cv::Rect rect : rectangles)
                cv::rectangle(workFrameMasked, rect, {0, 0, 255});

            writer.add(workFrameMasked, "Rectangles | squares=" + std::to_string(nRectangles) + " boxes=" + std::to_string(nBoxes));
//            cv::imshow("Rectangles", workFrame);
        }

        t.start();
        rectangles.clear();
        int nContours = 0;
        nContours += findRects(channels[0], rectangles);
        nContours += findRects(channels[1], rectangles);
        nContours += findRects(channels[2], rectangles);
        t.stop();
        std::cout << "[findRectangles]         time=" << t.get() << "ms nContours=" << nContours << " nRects" << rectangles.size() << std::endl;


        // ====================================================================================
        // ================================= CLUSTERING BEGIN =================================
        // ====================================================================================
        // Cluster rects by area
        SuperCluster <cv::Rect> scByArea;
        clusterRectsByArea(rectangles, scByArea);
        if(scByArea.empty())
            continue;

        // Select best cluster based on score
        t.start();
        int bestScore = 999999;
        int iBestCluster = 0;

        for(int i = 0; i < scByArea.size(); i++){
            const auto& cluster = scByArea[i];
//            std::cout << "[] i=" << i << " size=" << cluster.size() << std::endl;

            if(cluster.size() < 4)
                continue;
            if(100 < cluster.size())
                continue;

            std::vector<int> x;
            std::vector<int> y;

            for (const cv::Rect& rect : cluster) {
                x.emplace_back(rect.x);
                y.emplace_back(rect.y);
            }
            int varx = variance(x) / 100;
            int vary = variance(y) / 100;
            int score = (varx * vary) / cluster.size();
            if(score < bestScore){
                bestScore = score;
                iBestCluster = i;
            }
            std::cout << "varx=" << varx << " vary=" << vary << " score=" << score << std::endl;
        }
        t.stop();
        std::cout << "[selectClusterByScore]   time=" << t.get() << " score=" << bestScore << std::endl;
        // ====================================================================================
        // Filter duplicate rects of largest cluster
        Cluster<cv::Rect> cByArea;
        deduplicateRects(scByArea[iBestCluster], cByArea);
//        deduplicateRects(scByArea.back(), cByArea);

        // Draw all rects in scByArea
        workFrameClustering = cv::Mat::zeros(frame.rows, frame.cols, CV_8UC3);
        for(const auto& cluster : scByArea){
            for(int iRect = 0; iRect < cluster.size(); iRect++) {
                double iColour = 50 + (200 * iRect / cluster.size());
                cv::rectangle(workFrameClustering, cluster[iRect], {iColour, 0, iColour});
            }
        }
        // Draw largest of scByArea after its deduplication
        for (const cv::Rect& rect : cByArea)
            cv::rectangle(workFrameClustering, rect, {0, 255, 0}, 3);

        // sigh
        std::string strNClusters = std::to_string(scByArea.size()); strNClusters.resize(3, ' ');
        std::string strScore = std::to_string(bestScore); strScore.resize(3, ' ');
        std::string strArea = std::to_string(cByArea.front().area()); strArea.resize(3, ' ');
        writer.add(workFrameClustering, "Rects by area | nC=" + strNClusters + " score=" + strScore + " area=" + strArea);

//        cv::imshow("chosen", workFrameClustering);

        if(false){
        for(int i = 0; i < scByArea.size(); i++){
            if(scByArea[i].size() < 4)
                continue;
            if(100 < scByArea[i].size())
                continue;

            int factor = 4;
            workFrameClustering = cv::Mat::zeros(frame.rows/factor, frame.cols/factor, CV_8UC3);
            std::vector<int> x;
            std::vector<int> y;

            for (const cv::Rect& rect : scByArea[i]) {
                x.emplace_back(rect.x);
                y.emplace_back(rect.y);
                cv::Rect rect2;
                rect2.x = rect.x / factor; rect2.y = rect.y / factor; rect2.width = rect.width / factor; rect2.height = rect.height / factor;
                cv::rectangle(workFrameClustering, rect2, {0, 255, 0}, 0);
                cv::putText(workFrameClustering, std::to_string(rect.area()), rect2.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5, {255, 255, 255});
            }
            int varx = variance(x) / 1000;
            int vary = variance(y) / 1000;
            int score = (varx * vary) / scByArea[i].size();
            std::string s = "";
            s += "by area " + std::to_string(i);
            s += " size=" + std::to_string(scByArea[i].size());
            s += " " + std::to_string(varx);
            s += "," + std::to_string(vary);
            s += " | "+ std::to_string(score);
//            cv::putText(workFrameClustering, s, {10, 10}, cv::FONT_HERSHEY_SIMPLEX, 0.5, {255, 0, 0});
            cv::imshow(s, workFrameClustering);
        }
            cv::waitKey(0);
            cv::destroyAllWindows();
        }

//        continue;

        // ====================================================================================
        // Cluster most common rects by distance
        SuperCluster<Line> scByDistance;
        clusterRectsByDistance(cByArea, scByDistance, 0.7, LINE_MAX_LENGTH);
        if(scByDistance.empty())
            continue;
        Cluster<Line> mostCommonLines = scByDistance.back();

        // ====================================================================================
        // select lines by calculating the lines-intersect ratio based on bounding box
        int iBestLines = selectLinesByIntersectRatio(scByDistance);
        Cluster<Line> linesByRatio = scByDistance[iBestLines];

        // === Draw all squares, lines, and selected lines ===
        if(true) {
            workFrameClustering = cv::Mat::zeros(frame.rows, frame.cols, CV_8UC3);
            // Draw most common squares after deduplication
            for (const cv::Rect& rect : cByArea)
                cv::rectangle(workFrameClustering, rect, {255, 255, 255}, 3);
            // Draw all lines
            for(int iCluster = 0; iCluster < scByDistance.size(); iCluster++) {
                const Cluster<Line>& lineCluster = scByDistance[iCluster];
                double iColour = 50 + (200 * iCluster / scByDistance.size());
                for (const Line &line : lineCluster) {
                    const cv::Rect &r1 = std::get<1>(line);
                    const cv::Rect &r2 = std::get<2>(line);
                    cv::Point p1(r1.x + r1.width / 2, r1.y + r1.height / 2);
                    cv::Point p2(r2.x + r2.width / 2, r2.y + r2.height / 2);
                    cv::line(workFrameClustering, p1, p2, {iColour, 0, iColour}, 1);
                }
            }
        }

        // ====================================================================================
        // Cluster lines by angle
        SuperCluster<Line> linesByAngle;
        clusterLinesByAngle(linesByRatio, linesByAngle);
        Cluster<Line> bestLines = linesByAngle.back();

//      === Draw all squares, lines, and selected lines ===
        if(true) {
            // Draw selected lines
            for (const Line &line : bestLines) {
                const cv::Rect &r1 = std::get<1>(line);
                const cv::Rect &r2 = std::get<2>(line);
                cv::Point p1(r1.x + r1.width / 2, r1.y + r1.height / 2);
                cv::Point p2(r2.x + r2.width / 2, r2.y + r2.height / 2);
                cv::rectangle(workFrameClustering, r1, {0, 255, 0}, 3);
                cv::rectangle(workFrameClustering, r2, {0, 255, 0}, 3);
                cv::line(workFrameClustering, p1, p2, {0, 255, 0}, 3);
            }
            writer.add(workFrameClustering, "Lines by ratio and angle |  scAngle=" + std::to_string(linesByAngle.size()) + " nLines=" + std::to_string(bestLines.size()));
        }


        // ====================================================================================
        // Calculate the average rectangle size by averaging the length of the lines
        t.start();
        double avgRectSize = 0;
        if(!bestLines.empty()){
            // Friendly reminder that line lengths are stored without sqrt!
            for(const Line &line : bestLines) avgRectSize += sqrt(std::get<0>(line));
            avgRectSize /= bestLines.size();
        }
        t.stop();
        std::cout << "[averageRectSize]        time=" << t.get() << " size=" << avgRectSize << std::endl;


        // ====================================================================================
        // Get distinct squares and average rect size from the selected lines
        Cluster<ExtendedRect> distinctRects;
        if(!bestLines.empty()){
            t.start();
            for (const Line &line : bestLines) {
                const cv::Rect& r1 = std::get<1>(line);
                const cv::Rect& r2 = std::get<2>(line);

                bool r1Found = false;
                bool r2Found = false;

                for (const ExtendedRect& rect : distinctRects) {
                    r1Found |= (r1.x == rect.x && r1.y == rect.y);
                    r2Found |= (r2.x == rect.x && r2.y == rect.y);
                }

                if(!r1Found) distinctRects.emplace_back(r1, avgRectSize, std::get<3>(line));
                if(!r2Found) distinctRects.emplace_back(r2, avgRectSize, std::get<3>(line));
            }
            t.stop();
            std::cout << "[LinesToDistinctRects]   time=" << t.get() << "ms " << (bestLines.size() * 2) << " -> " << distinctRects.size() << std::endl;
        }


        // ====================================================================================
        // Get bounding box of all the distinct rects
        cv::Rect boundingBox = getBoundingBox(distinctRects);

        // Draw distinct squares and bounding box
//        if(!distinctRects.empty()) {
//            workFrameClustering = cv::Mat::zeros(frame.rows, frame.cols, CV_8UC1);
//            for (const ExtendedRect &rect : distinctRects)
//                DrawRotatedRectangle(workFrameClustering, rect);
//            cv::rectangle(workFrameClustering, boundingBox, {255}, 3);
//        }

//        cv::cvtColor(workFrameClustering, workFrameClustering, cv::COLOR_BGR2GRAY);
        workFrame.copyTo(frame, workFrameClustering);
        writer.add(frame);




        // ====================================================================================
        // Rotate all squares around the center of the bounding box
        workFrameClustering = cv::Mat::zeros(frame.rows, frame.cols, CV_8UC3);
        double rotationAngle = 0;
        int gridWidth = 0;
        int gridHeight = 0;
        bestScore = 0;
        cv::Point bestPoint(0, 0);

        if(!distinctRects.empty()){
            // Get the bouding box of all the rectangles
            cv::Point center(boundingBox.x + boundingBox.width / 2, boundingBox.y + boundingBox.height / 2);
            // Rotate all the rectangles around the center of this bounding box
            // Just grab the first angle, should work fine;
            double angleNorm = normalizeAngle90(-distinctRects.front().angle);
            cv::Size size((int)avgRectSize, (int)avgRectSize);

            Cluster<cv::Rect> rotatedRects;
            for(const ExtendedRect& rect : distinctRects){
                cv::Point newPoint = rotatePointAroundPoint(rect.tl(), center, -angleNorm);
                rotatedRects.emplace_back(newPoint, size);
            }

            // Get the bounding box of the rotated squares
            // Normalize all rotated squares to position (0, 0);
            cv::Rect box = getBoundingBox(rotatedRects);
            for (cv::Rect& rect : rotatedRects) {
                rect.x -= box.x;
                rect.y -= box.y;
                cv::rectangle(workFrameClustering, rect, {0, 255, 0}, 3);
            }
            cv::rectangle(workFrameClustering, box, {0, 0, 255}, 3);

            // Get the side of the grid
            Cluster<cv::Point> gridPoints;
            for (cv::Rect& rect : rotatedRects) {
                int dx = (int)round(rect.x / avgRectSize);
                int dy = (int)round(rect.y / avgRectSize);
                gridWidth  = std::max(dx+1, gridWidth);
                gridHeight = std::max(dy+1, gridHeight);

                cv::Rect dRect(dx * avgRectSize, dy * avgRectSize, (int)avgRectSize, (int)avgRectSize);
                cv::rectangle(workFrameClustering, dRect, {255, 0, 0}, 1);
                gridPoints.emplace_back(dx, dy);
            }

            // Create and fill grid
            int grid[gridWidth * gridHeight];
            for(int i = 0; i < gridWidth * gridHeight; i++)
                grid[i] = 0;
            for(const cv::Point& p : gridPoints)
                grid[p.y * gridWidth + p.x] = 1;

            // Find optimal grid
            std::cout << "[OptimalCube] gridSize=" << gridWidth << "x" << gridHeight << std::endl;
            for(int x = 0; x < gridWidth; x++){
            for(int y = 0; y < gridHeight; y++){

                int score = 0;
                for(int dx = x; dx < x + CUBE_SIZE && dx < gridWidth; dx++){
                for(int dy = y; dy < y + CUBE_SIZE && dy < gridHeight; dy++){
                    score += grid[dy * gridWidth + dx];
                }
                }
                if(bestScore < score){
                    bestScore = score;
                    bestPoint.x = x;
                    bestPoint.y = y;
                }
//                std::cout << "[OptimalCube] at " << x << "," << y << " score=" << score << std::endl;
            }
            }

            // Reset frame
            frame_og.copyTo(frame);

            // Create rotatedRects
            Cluster<cv::Point> gridRects;
            for(int x = 0; x < CUBE_SIZE; x++) {
            for(int y = 0; y < CUBE_SIZE; y++) {
                cv::Point p(x, y);  // Create point in grid
                p += bestPoint;     // Move to best place
                p *= avgRectSize;   // Move from grid to real
                p += box.tl();      // Move back into bounding box;
                p = rotatePointAroundPoint(p, center, angleNorm); // Rotate back to original place
                // Draw
                ExtendedRect r(cv::Rect(p, size), avgRectSize, angleNorm);
                DrawRotatedRectangle(frame, r);
            }
            }
        }

        writer.add(workFrameClustering, "Rects rotated | grid=" + std::to_string(gridWidth) + "x" + std::to_string(gridHeight) + " score=" + std::to_string(bestScore));
        writer.add(frame);


        std::cout << std::endl;

    }

}