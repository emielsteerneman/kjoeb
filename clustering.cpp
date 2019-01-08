// TODO don't cluster based on any match. Cluster based on median / mean / variance / etc.

#include "clustering.h"
#include "utils.h"
#include "maths.h"

namespace sorting {
    bool rectsByArea(const cv::Rect& r1, const cv::Rect& r2){
        return r1.area() < r2.area();
    }

    template <class T>
    bool clustersBySize(const Cluster<T> &c1, const Cluster<T> &c2){
        return c1.size() < c2.size();
    }

}

namespace clustering {

    void rectsByArea(const Cluster <cv::Rect> &rects, SuperCluster<cv::Rect> &clusters){
        Timer t;
        t.start();

        for(const cv::Rect& rect : rects){
            bool found = false;
            for(auto& cluster : clusters){
                for(const auto& rectComp : cluster){
                    if(found) break;
                    if(maths::inRangeRel(rect.area(), rectComp.area(), 0.90)){    // Should be compared with average / median instead of all.
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

        std::sort(clusters.begin(), clusters.end(), sorting::rectsByArea);

        t.stop();
        std::cout << "[clusterRectsByArea]     time=" << t.get() << "ms rects=" << rects.size() << " clusters=" << clusters.size() << std::endl;
    }

    void rectsByDistance(const Cluster<cv::Rect> &rects, SuperCluster<Line> &clusters, double threshold, int maxDistance, int minDistance){
        Timer t;
        t.start();

        // Find all lines between all rects
        int max = maxDistance * maxDistance;    // because not using root
        int min = minDistance * minDistance;    // because not using root
        std::vector<Line> lines;
        for(int iRect = 0; iRect < rects.size(); iRect++) {
            for (int jRect = iRect + 1; jRect < rects.size(); jRect++) {
                int distance = maths::distance(rects[iRect], rects[jRect]);
                if(min <= distance && distance <= max) {
                    double angle = std::atan2(rects[iRect].y - rects[jRect].y, rects[iRect].x - rects[jRect].x);
                    lines.emplace_back(std::forward_as_tuple(
                            maths::distance(rects[iRect], rects[jRect]),
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
                    if(maths::inRangeRel(std::get<0>(line), std::get<0>(lineComp), threshold)){    // Should be compared with average / median instead of all.
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
        std::sort(clusters.begin(), clusters.end(), sorting::clustersBySize<Line>);

        t.stop();
        std::cout << "[clusterRectsByDistance] time=" << t.get() << "ms lines=" << lines.size() << " clusters=" << clusters.size() << std::endl;
    }

    void linesByAngle(const Cluster<Line>& lines, SuperCluster<Line> &clusters){
        Timer t;
        t.start();

        double pi_step = M_PI / 4;
        double pi_threshold = pi_step * 0.05;

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
                    bool isInRange = maths::inRangeAbs(angle, 0, pi_threshold)
                                     || maths::inRangeAbs(angle, pi_step * 1, pi_threshold)
                                     || maths::inRangeAbs(angle, pi_step * 2, pi_threshold)
                                     || maths::inRangeAbs(angle, pi_step * 3, pi_threshold)
                                     || maths::inRangeAbs(angle, pi_step * 4, pi_threshold);

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
        std::sort(clusters.begin(), clusters.end(), sorting::clustersBySize<Line>);

        t.stop();
        std::cout << "[clusterLinesByAngle]    time=" << t.get() << "ms lines=" << lines.size() << " clusters=" << clusters.size() << std::endl;
    }

}

namespace selecting {

    double linesIntersectRatio(const Cluster<Line>& cluster){
        int iIntersections = 0;

        // Compare each line with each other, increment intersections if needed
        for (int iLine = 0; iLine < cluster.size(); iLine++)
            for (int jLine = iLine + 1; jLine < cluster.size(); jLine++)
                if(maths::intersect(cluster[iLine], cluster[jLine]))
                    iIntersections++;

        // +0.5 to compensate for cluster/0 vs cluster/1
        return iIntersections == 0 ? cluster.size() + 0.5 : cluster.size() / (float)iIntersections;
    }

    // Find the cluster with the best lines-to-intersections ratio (lines/intersections)
    // This can be done backward, and stopping when cluster.size() < bestRatio
    int linesByIntersectRatio(const SuperCluster<Line>& clusters){
        Timer t;
        t.start();

        double bestRatio = 0;
        int iBestCluster = -1;

        // For each cluster, count the intersections based on bounding boxes of lines
        for(int i = clusters.size()-1; 0 <= i; i--){
            const auto& cluster = clusters[i];

            if(cluster.size() < bestRatio)
                continue;

            double ratio = linesIntersectRatio(cluster);

            if(bestRatio < ratio){
                bestRatio = ratio;
                iBestCluster = i;
            }
        }

        t.stop();
        std::cout << "[linesByIntersectRatio]  time=" << t.get() << "ms bestCluster=" << iBestCluster << " size =" << clusters[iBestCluster].size() << " ratio=" << bestRatio << std::endl;
        return iBestCluster;
    }

    double rectsVarianceScore(const Cluster<cv::Rect>& cluster){
        std::vector<int> x, y;

        for (const cv::Rect& rect : cluster) {
            x.emplace_back(rect.x);
            y.emplace_back(rect.y);
        }

        double variance_x = maths::variance(x) / 100;
        double variance_y = maths::variance(y) / 100;

        return (variance_x * variance_y) / cluster.size();
    }

    int rectsByVarianceScore(const SuperCluster<cv::Rect>& clusters, int minSize, int maxSize){
        Timer t;
        t.start();

        double bestScore = DBL_MAX;
        int iBestCluster = -1;

        for(int i = 0; i < clusters.size(); i++){
            const auto& cluster = clusters[i];

            if(cluster.size() < minSize)
                continue;
            if(maxSize < cluster.size())
                continue;

            double score = rectsVarianceScore(cluster);

            if(score < bestScore){
                bestScore = score;
                iBestCluster = i;
            }
        }

        t.stop();
        std::cout << "[rectsByVarianceScore]   time=" << t.get() << " score=" << bestScore << std::endl;
        return iBestCluster;
    }

}