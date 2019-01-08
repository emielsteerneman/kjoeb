#include "utils.h"

Timer::Timer(){};

void Timer::start(){
    t_begin = std::chrono::steady_clock::now();
}

void Timer::stop() {
    t_end   = std::chrono::steady_clock::now();
}

double Timer::get(){
    return std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_begin).count() / 1000.0;
}






VidWriter::VidWriter(std::string filename, int w, int h, int nW, int nH, int fps)
        : width(w), height(h), nWidth(nW), nHeight(nH), fps(fps) {
    subWidth = width / nWidth;
    subHeight = height / nHeight;
    bool writerOpened = writer.open(filename, cv::VideoWriter::fourcc('a', 'v', 'c', '1'), fps, {w, h}, true);
    if(!writerOpened)
        std::cout << "[VidWriter] Warning! Could not open stream!" << std::endl;
    frame = cv::Mat::zeros(height, width, CV_8UC3);
    reset();
}

void VidWriter::disable(){
    disabled = true;
}

void VidWriter::reset(){
    at = 0;
}

void VidWriter::add(const cv::Mat& subFrame, std::string text){
    if(disabled) return;
    Timer t;
    t.start();

    // Resize
    cv::resize(subFrame, workFrame, {width / nWidth, height / nHeight});
//
//        std::cout << " Adding frame " << at << " | " << text << std::endl;
    if(!writer.isOpened())
        std::cout << "[VidWriter] Warning! Stream not opened" << std::endl;
    if(nWidth * nHeight <= at)
        std::cout << "[VidWriter] Warning! at=" << at << std::endl;

    int toWidth = (at % nWidth) * (width/nWidth);
    int toHeight = (int)floor(at/nHeight) * (height/nHeight);
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

void VidWriter::show(){
    if(disabled) return;
    cv::imshow("VideoWriter", frame);
    cv::namedWindow("VideoWriter", cv::WND_PROP_FULLSCREEN);
    cv::setWindowProperty("VideoWriter", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
}

void VidWriter::flush(){
    if(disabled) return;
    writer.write(frame);
    reset();
}