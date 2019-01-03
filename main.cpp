#include <iostream>

#include <opencv2/opencv.hpp>
#include <vector>
#include <tuple>
#include <numeric>
#include <limits>

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
    cv::VideoWriter writer;
    cv::Mat frame;
    bool disabled = false;
public:
    VidWriter(std::string filename, int w, int h, int nW, int nH)
        : width(w), height(h), nWidth(nW), nHeight(nH) {
        subWidth = width / nWidth;
        subHeight = height / nHeight;
        bool writerOpened = writer.open(filename, cv::VideoWriter::fourcc('a', 'v', 'c', '1'), 10.0, {w, h}, true);
        if(!writerOpened)
            std::cout << "[VidWriter] Warning! Could not open stream!" << std::endl;
        reset();
    }
    void disable(){disabled = true;}
    void reset(){ frame = cv::Mat::zeros(height, width, CV_8UC3); at = 0;}
    void add(const cv::Mat& subFrame){
        if(disabled) return;
        std::cout << " Adding frame " << at << std::endl;
        if(!writer.isOpened())
            std::cout << "[VidWriter] Warning! Stream not opened" << std::endl;
        if(9 <= at)
            std::cout << "[VidWriter] Warning! at=" << at << std::endl;

        int toWidth = (at % 3) * (width/nWidth);
        int toHeight = (int)floor(at/3) * (height/nHeight);
        cv::Rect dstRect(toWidth, toHeight, subWidth, subHeight);
        subFrame.copyTo(frame(dstRect));
        at++;
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
}


// ====================================================================================
// ==================================== EPIC MATHS ====================================
// ====================================================================================

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

void clusterRectsByDistance(const std::vector<cv::Rect> &rects, SuperCluster<Line> &clusters, int maxDistance = std::numeric_limits<int>::max(), int minDistance = 0){
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
                if(inRangeRel(std::get<0>(line), std::get<0>(lineComp), 0.90)){    // Should be compared with average / median instead of all.
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

    // Now we can fill the rotated rectangle with our specified color
    cv::fillConvexPoly(image,vertices,4,color);
}

int main() {
    std::cout << "Hello, World!" << std::endl;
    std::cout << std::setprecision(2) << std::fixed;

//    g();

    Timer t;

    cv::VideoCapture cap;
    if(!cap.open(0)) {
        std::cout << "Could not open capture device" << std::endl;
        return 0;
    }

    VidWriter writer("/home/emiel/Desktop/_kjoeb.mp4", 1920, 1080, 3, 3);
    writer.disable();

    // Set some constants
    const int WIDTH  = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    const int HEIGHT = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);

    const int LINE_MAX_LENGTH = std::min(WIDTH, HEIGHT) / 3;
    const int RECT_MAX_AREA   = LINE_MAX_LENGTH * LINE_MAX_LENGTH;

    cv::Mat frame, workFrame, workFrameClustering, mask;
    cv::Mat channels[3];

    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;

    std::vector<cv::Rect> rectangles;

    int total_rects = 0;
    int total_frames = 0;

    for(;;){
        std::cout << std::endl;

        writer.flush();

        // Grab frame
        if(cv::waitKey(10) == 27 ) break; // esc

        // Clear the annoying camera buffer -> always get the latest frame
//        for(int i = 0; i < 5; i++)
//            cap.grab();

        cap >> frame;
//        frame = cv::imread("../checkers.png");

        if(frame.empty()) break;

        // Shrink frame
//        cv::rpaulpaulesize(frame, frame, {frame.cols * 1 / 2, frame.rows * 1 / 2});
        cv::flip(frame, frame, 1);

        cv::imshow("Original", frame);
        cv::moveWindow("Original", 0, 0);
        cv::namedWindow("Original", cv::WND_PROP_FULLSCREEN);
        cv::setWindowProperty("Original", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
        writer.add(frame);

        frame.copyTo(workFrame);
        cv::blur(workFrame , workFrame, {7, 7});

        // Cluster colours
        workFrame = workFrame / 64;
        workFrame *= 64;

//        cv::imshow("Colour reduction", workFrame);
//        cv::moveWindow("Colour reduction", WIDTH/2, 0);
//        cv::namedWindow("Colour reduction", cv::WND_PROP_FULLSCREEN);
//        cv::setWindowProperty("Colour reduction", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
//        writer.add(workFrame);

        // Laplacian edge detection
        t.start();
        cv::Laplacian(workFrame, workFrame, CV_8UC3, 3);
        cv::normalize(workFrame, workFrame, 0, 255, cv::NORM_MINMAX, CV_32FC1);
        cv::convertScaleAbs(workFrame, workFrame);
        t.stop();
        std::cout << "[laplacian]              time=" << t.get() << std::endl;

//        cv::imshow("Border detection", workFrame);
//        cv::moveWindow("Border detection", WIDTH, 0);
//        cv::namedWindow("Border detection", cv::WND_PROP_FULLSCREEN);
//        cv::setWindowProperty("Border detection", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
//        writer.add(workFrame);


        cv::split(workFrame, channels);
//        workFrameClustering = cv::Mat::zeros(frame.rows, frame.cols, CV_8UC3);
//        rectangles.clear();
//        findRects(channels[0], rectangles);
//        for(cv::Rect rect : rectangles)
//            cv::rectangle(workFrameClustering, rect, {255, 0, 0});
//
//        rectangles.clear();
//        findRects(channels[1], rectangles);
//        for(cv::Rect rect : rectangles)
//            cv::rectangle(workFrameClustering, rect, {0, 255, 0});
//
//        rectangles.clear();
//        findRects(channels[2], rectangles);
//        for(cv::Rect rect : rectangles)
//            cv::rectangle(workFrameClustering, rect, {0, 0, 255});
//
//        cv::imshow("Squares", workFrameClustering);
//        cv::moveWindow("Squares", 0, HEIGHT/2);
//        cv::namedWindow("Squares", cv::WND_PROP_FULLSCREEN);
//        cv::setWindowProperty("Squares", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
//        writer.add(workFrameClustering);

        rectangles.clear();
        findRects(channels[0], rectangles);
        findRects(channels[1], rectangles);
        findRects(channels[2], rectangles);


        // ====================================================================================
        // ================================= CLUSTERING BEGIN =================================
        // ====================================================================================
        // Cluster rects by area
        SuperCluster <cv::Rect> clusters;
        clusterRectsByArea(rectangles, clusters);
        if(clusters.empty())
            continue;

        // Draw clusters
//        for(int i = 0; i < clusters.size(); i++)
//            for(const cv::Rect& rect : clusters[i])
//                cv::rectangle(workFrameClustering, rect, {20, 20, 20},3);

        // Draw and Analyze largest cluster
//        cv::Scalar colours[] = {{255, 0, 0, 0.25}, {0, 255, 0, 0.25}, {0, 0, 255, 0.25}};
//        for (const cv::Rect& rect : clusters.back())
//            cv::rectangle(workFrameClustering, rect, {255, 255, 255}, 2);

        // ====================================================================================
        // Filter duplicate rects of largest cluster
        Cluster<cv::Rect> mostCommonRects;
        deduplicateRects(clusters.back(), mostCommonRects);

        // Draw all clusters
        workFrameClustering = cv::Mat::zeros(frame.rows, frame.cols, CV_8UC3);
        for(const auto& cluster : clusters){
            for(int iRect = 0; iRect < cluster.size(); iRect++) {
                double iColour = 50 + (200 * iRect / cluster.size());
                cv::rectangle(workFrameClustering, cluster[iRect], {iColour, 0, iColour});
            }
        }

        // Draw most common squares after deduplication
        for (const cv::Rect& rect : mostCommonRects)
            cv::rectangle(workFrameClustering, rect, {0, 255, 0}, 3);

//        cv::imshow("clusterRectsByArea", workFrameClustering);
//        cv::moveWindow("clusterRectsByArea", WIDTH/2, HEIGHT/2);
//        cv::namedWindow("clusterRectsByArea", cv::WND_PROP_FULLSCREEN);
//        cv::setWindowProperty("clusterRectsByArea", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
//        writer.add(workFrameClustering);

        // ====================================================================================
        // Cluster most common rects by distance
        SuperCluster<Line> linesByDistance;
        clusterRectsByDistance(mostCommonRects, linesByDistance, LINE_MAX_LENGTH);
        if(linesByDistance.empty())
            continue;
        Cluster<Line> mostCommonLines = linesByDistance.back();

        // ====================================================================================
        // select lines by calculating the lines-intersect ratio based on bounding box
        int iBestLines = selectLinesByIntersectRatio(linesByDistance);
        Cluster<Line> linesByRatio = linesByDistance[iBestLines];

        // === Draw all squares, lines, and selected lines ===
        // Draw most common squares after deduplication
//        workFrameClustering = cv::Mat::zeros(frame.rows, frame.cols, CV_8UC3);
//        for (const cv::Rect& rect : mostCommonRects)
//            cv::rectangle(workFrameClustering, rect, {255, 255, 255}, 3);
//        // Draw lines
//        for(int iCluster = 0; iCluster < linesByDistance.size(); iCluster++) {
//            const Cluster<Line>& lineCluster = linesByDistance[iCluster];
//            double iColour = 50 + (200 * iCluster / linesByDistance.size());
//            for (const Line &line : lineCluster) {
//                const cv::Rect &r1 = std::get<1>(line);
//                const cv::Rect &r2 = std::get<2>(line);
//                cv::Point p1(r1.x + r1.width / 2, r1.y + r1.height / 2);
//                cv::Point p2(r2.x + r2.width / 2, r2.y + r2.height / 2);
//                cv::line(workFrameClustering, p1, p2, {iColour, 0, iColour}, 1);
//            }
//        }
//        // Draw selected lines
//        {
//            for (const Line &line : linesByRatio) {
//                const cv::Rect &r1 = std::get<1>(line);
//                const cv::Rect &r2 = std::get<2>(line);
//                cv::Point p1(r1.x + r1.width / 2, r1.y + r1.height / 2);
//                cv::Point p2(r2.x + r2.width / 2, r2.y + r2.height / 2);
//                cv::line(workFrameClustering, p1, p2, {0, 255, 0}, 3);
//            }
//        }
//        cv::imshow("selectLinesByIntersectRatio", workFrameClustering);
//        cv::moveWindow("selectLinesByIntersectRatio", WIDTH, HEIGHT/2);
//        cv::namedWindow("selectLinesByIntersectRatio", cv::WND_PROP_FULLSCREEN);
//        cv::setWindowProperty("selectLinesByIntersectRatio", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
//        writer.add(workFrameClustering);

        // ====================================================================================
        // Cluster lines by angle
        SuperCluster<Line> linesByAngle;
        clusterLinesByAngle(linesByRatio, linesByAngle);

        Cluster<Line> bestLines = linesByAngle.back();


//         === Draw all squares, lines, and selected lines ===
//         Draw most common squares after deduplication
//        workFrameClustering = cv::Mat::zeros(frame.rows, frame.cols, CV_8UC3);
//        // Draw lines
//        {
//            for (const Line &line : linesByRatio) {
//                const cv::Rect &r1 = std::get<1>(line);
//                const cv::Rect &r2 = std::get<2>(line);
//                cv::Point p1(r1.x + r1.width / 2, r1.y + r1.height / 2);
//                cv::Point p2(r2.x + r2.width / 2, r2.y + r2.height / 2);
//                cv::line(workFrameClustering, p1, p2, {255, 255, 255}, 3);
//            }
//        }
//        // Draw selected lines
//        for (const Line &line : bestLines) {
//            const cv::Rect &r1 = std::get<1>(line);
//            const cv::Rect &r2 = std::get<2>(line);
//            cv::Point p1(r1.x + r1.width / 2, r1.y + r1.height / 2);
//            cv::Point p2(r2.x + r2.width / 2, r2.y + r2.height / 2);
//            cv::rectangle(workFrameClustering, r1, {0, 255, 0}, 3);
//            cv::rectangle(workFrameClustering, r2, {0, 255, 0}, 3);
//            cv::line(workFrameClustering, p1, p2, {0, 255, 0}, 3);
//        }
//        cv::imshow("clusterLinesByAngle", workFrameClustering);
//        cv::moveWindow("clusterLinesByAngle", 0, HEIGHT);
//        cv::namedWindow("clusterLinesByAngle", cv::WND_PROP_FULLSCREEN);
//        cv::setWindowProperty("clusterLinesByAngle", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
//        writer.add(workFrameClustering);



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
        cv::Rect boundingBox;
        if(!distinctRects.empty()){
            int xMin = frame.cols;
            int xMax = 0;
            int yMin = frame.rows;
            int yMax = 0;
            for(const ExtendedRect& rect : distinctRects){
                xMin = std::min(xMin, rect.x);
                xMax = std::max(xMax, rect.x + rect.width);
                yMin = std::min(yMin, rect.y);
                yMax = std::max(yMax, rect.y + rect.height);
            }
            boundingBox.x = xMin;
            boundingBox.width = xMax - xMin;
            boundingBox.y = yMin;
            boundingBox.height = yMax - yMin;
            std::cout << "[BoundingBox]            xMin=" << xMin << " xMax=" << xMax << " yMin=" << yMin << " yMax=" << yMax << std::endl;
        }


        // Draw distinct squares and bounding box
        if(!distinctRects.empty()) {
            workFrameClustering = cv::Mat::zeros(frame.rows, frame.cols, CV_8UC3);
            for (const ExtendedRect &rect : distinctRects)
                DrawRotatedRectangle(workFrameClustering, rect);
            cv::rectangle(workFrameClustering, boundingBox, {255, 255, 255}, 3);
        }
//        cv::imshow("DistinctSquares", workFrameClustering);
//        cv::moveWindow("DistinctSquares", WIDTH/2, HEIGHT);
//        cv::namedWindow("DistinctSquares", cv::WND_PROP_FULLSCREEN);
//        cv::setWindowProperty("DistinctSquares", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
//        writer.add(workFrameClustering);


//        cv::cvtColor(workFrameClustering, workFrameClustering, cv::COLOR_BGR2GRAY);
        workFrame.copyTo(frame, workFrameClustering);
        cv::imshow("Result", frame);
        cv::moveWindow("Result", WIDTH/2, 0);
        cv::namedWindow("Result", cv::WND_PROP_FULLSCREEN);
        cv::setWindowProperty("Result", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
        writer.add(frame);



        // ====================================================================================
        // Rotate all squares around the center of the bounding box
        if(!distinctRects.empty()){
            Cluster<cv::Rect> rotatedRects;

//            cv::Point center(boundingBox.x + boundingBox.width, boundingBox.y + boundingBox.height);
            cv::Point center(frame.cols / 2, frame.rows/2);
            double angle = -distinctRects.front().angle; // Just grab the first angle, should work fine;
            cv::Size size(avgRectSize, avgRectSize);

            for(const ExtendedRect& rect : distinctRects){
                cv::Point newPoint = rotatePointAroundPoint(rect.tl(), center, angle);
                rotatedRects.emplace_back(newPoint, size);
            }

            workFrameClustering = cv::Mat::zeros(frame.rows, frame.cols, CV_8UC3);
            for (const cv::Rect& rect : rotatedRects)
                cv::rectangle(workFrameClustering, rect, {0, 255, 0}, 3);
            cv::rectangle(workFrameClustering, boundingBox, {255, 255, 255}, 3);
        }

        cv::imshow("RotatedSquares", workFrameClustering);
        cv::moveWindow("RotatedSquares", WIDTH*3/2, 0);
        cv::namedWindow("RotatedSquares", cv::WND_PROP_FULLSCREEN);
        cv::setWindowProperty("RotatedSquares", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
        writer.add(workFrameClustering);






        std::cout << std::endl;
        continue;


    }

}