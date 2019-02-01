// TODO don't cluster based on any match. Cluster based on median / mean / variance / etc.

#include "clustering.h"
#include "utils.h"
#include "maths.h"

namespace sorting {

    template <class T>
    bool clustersBySize(const Cluster<T> &c1, const Cluster<T> &c2){
        return c1.size() < c2.size();
    }

}

namespace clustering {

    void rectsByArea(const Cluster<ExtendedRect>& rects, SuperCluster<ExtendedRect>& clusters, double threshold){
        Timer t;
        t.start();

        for(const ExtendedRect& rect : rects){
            bool found = false;
            for(auto& cluster : clusters){
                for(const auto& rectComp : cluster){
                    if(found) break;
                    if(maths::inRangeRel(rect.area(), rectComp.area(), threshold)){    // Should be compared with average / median instead of all.
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

        std::sort(clusters.begin(), clusters.end());

        t.stop();
        std::cout << "[clusterRectsByArea]     time=" << t.get() << "ms rects=" << rects.size() << " clusters=" << clusters.size() << std::endl;
    }

    void rectsByDistance(Cluster<ExtendedRect> &rects, SuperCluster<Line> &clusters, double threshold, int maxDistance, int minDistance){
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
                    lines.emplace_back(
                        maths::distance(rects[iRect], rects[jRect]),
                        angle,
                        rects[iRect],
                        rects[jRect]
                    );
                }
            }
        }

        // Cluster lines by length
        for(const Line& line : lines){
            bool found = false;
            for(auto& cluster : clusters){
                for(const auto& lineComp : cluster){
                    if(found) break;
                    if(maths::inRangeRel(line.distance, lineComp.distance, threshold)){    // Should be compared with average / median instead of all.
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
        std::cout << "[clusterRectsByDistance] time=" << t.get() << "ms rects=" << rects.size() << " lines=" << lines.size() << " clusters=" << clusters.size() << std::endl;
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

                    double a1 = std::atan2(line.r1.y     - line.r2.y    , line.r1.x     - line.r2.x);
                    double a2 = std::atan2(lineComp.r1.y - lineComp.r2.y, lineComp.r1.x - lineComp.r2.x);

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

    double rectsVarianceScore(const Cluster<ExtendedRect>& cluster){
        std::vector<int> x, y;

        for (const ExtendedRect& rect : cluster) {
            x.emplace_back(rect.x);
            y.emplace_back(rect.y);
        }

        double variance_x = maths::variance(x) / 100;
        double variance_y = maths::variance(y) / 100;

//        std::cout << "[rectsVarianceScore]       varx=" << variance_x << " vary=" << variance_y << " score=" << ((variance_x * variance_y) / cluster.size()) << std::endl;

        return (variance_x * variance_y) / cluster.size();
    }

    int rectsByVarianceScore(const SuperCluster<ExtendedRect>& clusters, int minSize, int maxSize){
        Timer t;
        t.start();

        double bestScore = DBL_MAX;
        int iBestCluster = -1;

        for(int i = 0; i < clusters.size(); i++){
            const auto& cluster = clusters[i];

//            std::cout << "[rectsByVarianceScore]   Cluster " << i << " with size " << cluster.size() << std::endl;

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