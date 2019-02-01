#include "utils.h"
#include "clustering.h"
#include "drawing.h"

#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <tuple>
#include <numeric>
#include <limits>
#include "maths.h"

const int CUBE_SIZE = 4;
const int FRAME_WIDTH = 1280;
const int FRAME_HEIGHT = 720;

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



int findRects(cv::Mat& img, Cluster<ExtendedRect>& rects, int minArea = 0, int maxArea = 999999){
    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;

    findContours(img, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

    for(const auto& contour : contours) {
        cv::Rect box = cv::boundingRect(contour);
        if(minArea <= box.area() && box.area() <= maxArea)
            if(maths::inRangeRel(box.width, box.height))
                rects.push_back(box);
    }
    return contours.size();
}


// ====================================================================================
// ==================================== EPIC MATHS ====================================
// ====================================================================================


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








// Rects are duplicate if the distance between their centers are small compared to their sizes
void deduplicateRects(const Cluster<ExtendedRect>& in, Cluster<ExtendedRect>& out){
    Timer t;
    t.start();

    for(const ExtendedRect& rect : in){
        bool found = false;
        for(const cv::Rect& rectComp : out){
            double dist = maths::distance(rect, rectComp);
            if(dist < 0.25 * rect.area()) {
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

    const int FPS = 5;

    Timer t;
    Timer tTotal;

    cv::VideoCapture cap;
    if(!cap.open(0)) {
        std::cout << "Could not open capture device" << std::endl;
        return 0;
    }

//    if(!cap.open("/home/emiel/kjoebbbb.mkv")) {
//        std::cout << "Could not open video" << std::endl;
//        return 0;
//    }

    cap.grab();

    const int LINE_MAX_LENGTH = std::min(FRAME_WIDTH, FRAME_HEIGHT) / CUBE_SIZE;
    const int LINE_MIN_LENGTH = std::min(FRAME_WIDTH, FRAME_HEIGHT) / (CUBE_SIZE * 4);
    const int RECT_MAX_AREA   = LINE_MAX_LENGTH * LINE_MAX_LENGTH;
    const int RECT_MIN_AREA   = LINE_MIN_LENGTH * LINE_MIN_LENGTH;

//    std::cout << "CAP_PROP_FRAME_WIDTH  " << cap.get(cv::CAP_PROP_FRAME_WIDTH) << std::endl;
//    std::cout << "CAP_PROP_FRAME_HEIGHT " << cap.get(cv::CAP_PROP_FRAME_HEIGHT) << std::endl;
//    std::cout << "CAP_PROP_FPS          " << cap.get(cv::CAP_PROP_FPS) << std::endl;
//
//    if(!cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G')))
//        std::cout << "Could not set CAP_PROP_FOURCC" << std::endl;
//
//    if(!cap.set(cv::CAP_PROP_FRAME_WIDTH, FRAME_WIDTH))
//        std::cout << "Could not set CAP_PROP_FRAME_WIDTH" << std::endl;
//
//    if(!cap.set(cv::CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT))
//        std::cout << "Could not set CAP_PROP_FRAME_HEIGHT" << std::endl;
//
//    if(!cap.set(cv::CAP_PROP_FPS, 60))
//        std::cout << "Could not set CAP_PROP_FPS" << std::endl;

    std::cout << "RECT_MIN_AREA         " << RECT_MIN_AREA << std::endl;
    std::cout << "RECT_MAX_AREA         " << RECT_MAX_AREA << std::endl;

    cv::Mat frame, frame_og, frameMask, workFrame, workFrameClustering, imshowFrame, mask;
    cv::Mat channels[3];

    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;
    Cluster<ExtendedRect> rectangles;

    int erosion_size = 1;
    cv::Mat element = getStructuringElement(cv::MORPH_CROSS,
            cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1),
            cv::Point(erosion_size, erosion_size) );

    VidWriter writer("/home/emiel/Desktop/kjoeb/kjoeb.mp4", 1600, 900, 3, 3, 1000 / 65);
//    VidWriter writer("/home/emiel/Desktop/kjoeb/kjoeb.mp4", 1920, 1080, 3, 3, 1000 / 65);

//    writer.disable();

    int nFrames = 0;

    std::cout << (frameMask.empty() ? "Empty!" : "Not empty!") << std::endl;

    int delay = 0;
    int timeout = 10;

    cap.set(cv::CAP_PROP_POS_FRAMES, 180); nFrames = 180;

    for(;;) {
        std::cout << std::endl;
        std::cout << "=== IMAGE PROCESSING ====" << std::endl;

        nFrames++;

        writer.show();
        writer.flush();
        writer.reset();

        int key = cv::waitKey(delay);
        if(key == 27)
            break;
        if(key == 32) // space
            delay += timeout - 2 * delay;
        if(key == 81){
            delay = 0;
            cap.set(cv::CAP_PROP_POS_FRAMES, cap.get(cv::CAP_PROP_POS_FRAMES) - 2);
            nFrames -= 2;
        }

        // === Capture frame
        t.start();
        bool isCaptured = cap.read(frame_og);
        frame_og.copyTo(frame);
//        cv::resize(frame, frame, {640, 360});
        t.stop();
        std::cout << "[capture]                time=" << t.get() << "ms nFrames=" << nFrames << " res=" << frame_og.cols
                  << "x" << frame_og.rows << std::endl;

        // === Skip empty frames
        if(frame.empty()){
            std::cout << "Frame is empty!" << std::endl;
            cap.set(cv::CAP_PROP_POS_FRAMES, 0);
            nFrames = 0;
            continue;
        }

        // Shrink frame
        // cv::resize(frame, frame, {FRAME_WIDTH, FRAME_HEIGHT});
        tTotal.stop();
        writer.add(frame, "Original Frame | @time=" + std::to_string(tTotal.now()) + "  " + std::to_string(nFrames) + " " + std::to_string(tTotal.get()) + "ms");
        std::cout << "[TotalTime]              total time=" << tTotal.get() << "ms" << std::endl;
        tTotal.start();

        cv::GaussianBlur(frame, workFrame, cv::Size(0, 0), 9);
        cv::addWeighted(frame, 2.0, workFrame, -1.0, 0, workFrame);
        writer.add(workFrame, "Sharpened Frame");
        workFrame.copyTo(frame);

        // === Canny Edge Detection === //
        t.start();
        Canny(frame, workFrame, 60, 180, 3);
        cv::dilate(workFrame, workFrame, element);
        cv::dilate(workFrame, workFrame, element);
        workFrame = 255 - workFrame;

        // === Apply frame mask === //
        if(!frameMask.empty())
            workFrame &= frameMask;

        // === Find rectangles === //
        int nContours = 0;
        int nRectangles = 0;
        rectangles.clear();

        nContours += findRects(workFrame, rectangles, RECT_MIN_AREA, RECT_MAX_AREA);
        nRectangles += rectangles.size();

        // Sort rectangles by area
        std::sort(rectangles.begin(), rectangles.end());

        for(const cv::Rect& r : rectangles){
            std::cout << r.area() << std::endl;
        }


        // Give each rectangle its own unique id
        for(int id = 0; id < rectangles.size(); id++)
            rectangles[id].id = id;
        t.stop();
        std::cout << "[findRectangles]         time=" << t.get() << "ms nContours=" << nContours << " nRects" << rectangles.size() << std::endl;

        // Draw all rectangles found on top of the Canny edge detection
        cv::cvtColor(workFrame, workFrame, cv::COLOR_GRAY2BGR);
        for(cv::Rect rect : rectangles)
            cv::rectangle(workFrame, rect, {0, 255, 0});
        writer.add(workFrame, "Rectangles | @time=" + std::to_string(tTotal.now()) + " " + std::to_string(nRectangles) + " / " + std::to_string(nContours));

        // ====================================================================================
        // ================================= CLUSTERING BEGIN =================================
        // ====================================================================================
        std::cout << "=== CLUSTERING ==========" << std::endl;

        /// Cluster rects by area
        SuperCluster<ExtendedRect> scByArea;
        clustering::rectsByArea(rectangles, scByArea, 0.7);
        /// Select best cluster by variance score
        int iBestCluster = selecting::rectsByVarianceScore(scByArea, CUBE_SIZE, CUBE_SIZE * CUBE_SIZE * 3);
        if(iBestCluster == -1){
            std::cout << "[rectsByVarianceScore]   No suitable cluster found!" << std::endl;
            continue;
        }

        /// Deduplicate rects of best cluster
        Cluster<ExtendedRect> cByArea;
        deduplicateRects(scByArea[iBestCluster], cByArea);
        if(cByArea.empty()) continue;

        // Draw all clusters in scByArea in purple, chosen in green
        if(true) {
            imshowFrame = cv::Mat::zeros(frame.rows, frame.cols, CV_8UC3);
            drawSuperCluster(imshowFrame, scByArea);
            drawCluster(imshowFrame, cByArea, {0, 255, 0}, 3);
            writer.add(imshowFrame, "By area @time=" + std::to_string(tTotal.now()) + " nC=" + std::to_string(scByArea.size()) + " area=" + std::to_string(cByArea.front().area()));
        }

        /// Cluster rects by distance -> lines
        SuperCluster<Line> scByDistance;
        clustering::rectsByDistance(cByArea, scByDistance, 0.8, LINE_MAX_LENGTH);
        if(scByDistance.empty()){
            std::cout << "[rectsByDistance]        No suitable clusters found!" << std::endl;
            continue;
        }
        /// Select lines by calculating the lines-intersect ratio based on bounding box
        int iBestLines = selecting::linesByIntersectRatio(scByDistance);
        if(iBestLines == -1){
            std::cout << "[linesByIntersectRatio]  No cluster has the best score!" << std::endl;
            continue;
        }
        Cluster<Line> linesByDistance = scByDistance[iBestLines];

        /// Cluster lines by angle
        SuperCluster<Line> linesByAngle;
        clustering::linesByAngle(linesByDistance, linesByAngle);
        Cluster<Line> bestLines = linesByAngle.back();

        // === Draw all rectangles, lines, and selected lines ===
        if(true) {
            imshowFrame = cv::Mat::zeros(frame.rows, frame.cols, CV_8UC3);
            // Draw rectangles by area in white
            drawCluster(imshowFrame, cByArea, {255, 255, 255});
            // Draw rectangles by distance in green
            drawCluster(imshowFrame, linesByDistance, {0, 0, 255});
            // Draw rectangles by angle in red
            drawCluster(imshowFrame, bestLines, {0, 255, 0});
            writer.add(imshowFrame, "Lines @time=" + std::to_string(tTotal.now()) + " by distance and angle");
        }

        // Imshow all individual clusters and their scores
        if(false){
            int factor = 1;
            for(int i = 0; i < scByArea.size(); i++){
                if(scByArea[i].size() < 4)
                    continue;
                if(100 < scByArea[i].size())
                    continue;

                imshowFrame = cv::Mat::zeros(frame.rows/factor, frame.cols/factor, CV_8UC3);
                for (const cv::Rect& rect : scByArea[i]) {
                    cv::Rect rect2;
                    rect2.x = rect.x / factor; rect2.y = rect.y / factor; rect2.width = rect.width / factor; rect2.height = rect.height / factor;
                    cv::rectangle(imshowFrame, rect2, {0, 255, 0}, 0);
                    cv::putText(imshowFrame, std::to_string(rect.area()), rect2.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5, {255, 255, 255});
                }
                double score = selecting::rectsVarianceScore(scByArea[i]);
                std::string s;
                s += "by area " + std::to_string(i);
                s += " size=" + std::to_string(scByArea[i].size());
                s += " | "+ std::to_string(score);
                cv::imshow(s, imshowFrame);
            }
            writer.show();
            if(cv::waitKey(0) == 27 ) break; // esc
            cv::destroyAllWindows();
            continue;
        }

        // Show all line clusters in a separate window
        if(false){
            // Draw all lines
            for(int iCluster = 0; iCluster < scByDistance.size(); iCluster++) {
                const Cluster<Line>& lineCluster = scByDistance[iCluster];
                imshowFrame = cv::Mat::zeros(frame.rows, frame.cols, CV_8UC3);
                for (const Line &line : lineCluster) {
                    cv::Point p1(line.r1.x + line.r1.width / 2, line.r1.y + line.r1.height / 2);
                    cv::Point p2(line.r2.x + line.r2.width / 2, line.r2.y + line.r2.height / 2);
                    cv::line(imshowFrame, p1, p2, {255, 255, 255}, 1);
                }

                std::string s = "";
                s += "lines " + std::to_string(iCluster);
                s += " size=" + std::to_string(lineCluster.size());
                cv::imshow(s, imshowFrame);
            }
            writer.show();
            if(cv::waitKey(0) == 27 ) break; // esc
            cv::destroyAllWindows();
            continue;
        }

        // ====================================================================================
        // ================================== GRIDDING BEGIN ==================================
        // ====================================================================================
        std::cout << "=== GRIDDING ============" << std::endl;

        if(bestLines.empty()){
            std::cout << "[bestLines]              No lines to create grid from!" << std::endl;
            continue;
        }


        // Calculate the average rectangle size by averaging the length of the lines
        // Also calculate the average angle
        t.start();
        double rectSizeNorm = 0.0;
        double rectAngleNorm = 0.0;
        double angleOffset = bestLines.front().angle;
        // Friendly reminder that line lengths are stored without sqrt!
        for(const Line &line : bestLines) {
            rectSizeNorm += sqrt(line.distance);
            rectAngleNorm += maths::normalizeAngle90(line.angle - angleOffset);
        }
        rectSizeNorm /= bestLines.size();
        rectAngleNorm = rectAngleNorm / bestLines.size() + angleOffset;
        t.stop();
        std::cout << "[averageRectSizeAngle]   time=" << t.get() << " size=" << rectSizeNorm << " angle=" << rectAngleNorm << std::endl;


        // Get distinct rectangles from the selected lines
        Cluster<ExtendedRect> distinctRects;
        t.start();
        for (const Line &line : bestLines) {

            bool r1Found = false;
            bool r2Found = false;

            for (const ExtendedRect& rect : distinctRects) {
                r1Found |= line.r1.id == rect.id;
                r2Found |= line.r2.id == rect.id;
            }

            if(!r1Found) distinctRects.push_back(line.r1);
            if(!r2Found) distinctRects.push_back(line.r2);
        }
        t.stop();
        std::cout << "[LinesToDistinctRects]   time=" << t.get() << "ms " << (bestLines.size() * 2) << " -> " << distinctRects.size() << std::endl;

        // Set size and angle for each rectangle
        for(ExtendedRect& rect : distinctRects) {
            rect.sizeNorm = rectSizeNorm;
            rect.angle = rectAngleNorm;
        }

        // Get bounding box fo all rectangles
        cv::Rect boundingBox = getBoundingBox(distinctRects);

        // === Draw distinct rectangles and bounding box ===
        if(true) {
            imshowFrame = cv::Mat::zeros(frame.rows, frame.cols, CV_8UC3);
            drawCluster(imshowFrame, distinctRects, {0, 255, 0});
            cv::rectangle(imshowFrame, boundingBox, {0, 255, 0}, 2);

            for(const ExtendedRect& rect : distinctRects)
                DrawRotatedRectangle(imshowFrame, rect);

            writer.add(imshowFrame, "Bounding box @time=" + std::to_string(tTotal.now()));
        }


        // === Rotate all rects around the center of the bounding box === //
        // Get center of the bouding box of all the rectangles
        cv::Point gridCenter(boundingBox.x + boundingBox.width / 2, boundingBox.y + boundingBox.height / 2);
        for(ExtendedRect& rect : distinctRects){
            cv::Point newPoint = rotatePointAroundPoint(rect.tl(), gridCenter, - rectAngleNorm);
            rect.x = newPoint.x;
            rect.y = newPoint.y;
        }

        // === Get the bounding box of the rotated rects, and normalize all rotated rects to position (0, 0); === //
        cv::Rect box = getBoundingBox(distinctRects);
        for (ExtendedRect& rect : distinctRects) {
            rect.x -= box.x;
            rect.y -= box.y;
        }


        // === Draw rotated rectangles ===
        if(true) {
            imshowFrame = cv::Mat::zeros(frame.rows, frame.cols, CV_8UC3);
//            cv::rectangle(imshowFrame, box, {0, 255, 0}, 2);
            int tx = 0;
            int ty = 0;
            for(const ExtendedRect& rect : distinctRects) {
                ExtendedRect rect2(rect);
                tx += rect.x;
                ty += rect.y;
                rect2.x = rect.x + 40;
                rect2.y = rect.y + 40;
                rect2.angle = rect.angle - rectAngleNorm;
                rect2.sizeNorm = rect.sizeNorm;
                DrawRotatedRectangle(imshowFrame, rect2);
            }
            tx /= distinctRects.size();
            ty /= distinctRects.size();
            cv::Rect r(tx - 10, ty - 10, 20, 20);
            r.x += 40 + rectSizeNorm/2; r.y += 40 + rectSizeNorm/2;
            cv::rectangle(imshowFrame, r, {0, 0, 255}, 3);
            writer.add(imshowFrame, "Rotated rectangles @time=" + std::to_string(tTotal.now()));
        }



        // === Calculate the size of the grid === //
        int gridWidth = 0;
        int gridHeight = 0;
        Cluster<cv::Point> gridPoints;
        for(ExtendedRect& rect : distinctRects) {
            int dx = (int)round(rect.x / rectSizeNorm);
            int dy = (int)round(rect.y / rectSizeNorm);
            gridWidth  = std::max(dx+1, gridWidth);
            gridHeight = std::max(dy+1, gridHeight);
            gridPoints.emplace_back(dx, dy);
        }

        // === Create and fill grid === //
        int grid[gridWidth * gridHeight];
        for(int i = 0; i < gridWidth * gridHeight; i++)
            grid[i] = 0;
        for(const cv::Point& p : gridPoints)
            grid[p.y * gridWidth + p.x] = 1;

        // === Find the optimal grid === //
        int bestScore = 0;
        cv::Point bestPoint(0, 0);

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
        }
        }

        // Create new rects
        Cluster<ExtendedRect> gridRects;
        for(int x = 0; x < CUBE_SIZE; x++) {
        for(int y = 0; y < CUBE_SIZE; y++) {
            cv::Point p(x, y);  // Create point in grid
            p += bestPoint;     // Move to best place
            p *= rectSizeNorm;  // Move from grid to real
            p += box.tl();      // Move back into bounding box;
            p = rotatePointAroundPoint(p, gridCenter, rectAngleNorm); // Rotate back to original place

            cv::Rect r(p, cv::Size(rectSizeNorm, rectSizeNorm));
            gridRects.emplace_back(r, y * CUBE_SIZE + x, rectSizeNorm, rectAngleNorm);
        }
        }

        if(true){
            frame.copyTo(imshowFrame);
//            imshowFrame = cv::Mat::zeros(frame.rows, frame.cols, CV_8UC3);
            for(const ExtendedRect& rect : gridRects)
                DrawRotatedRectangle(imshowFrame, rect);
            writer.add(imshowFrame, "Grid @time=" + std::to_string(tTotal.now()));
        }

        if(CUBE_SIZE * CUBE_SIZE - CUBE_SIZE <= bestScore){
            frameMask = cv::Mat::zeros(frame.rows, frame.cols, CV_8UC1);
            cv::Point p1 = boundingBox.tl() - cv::Point(boundingBox.width/4, boundingBox.width/4);
            cv::Point p2 = boundingBox.br() + cv::Point(boundingBox.width/4, boundingBox.width/4);

            cv::rectangle(frameMask, p1, p2, 255, -1);
//            cv::imshow("frameMask", frameMask);
//            cv::waitKey(0);
//            cv::destroyAllWindows();
        }else{

        }

        std::cout << std::endl;
    }
}