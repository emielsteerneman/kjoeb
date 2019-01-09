//
// Created by emiel on 8-1-19.
//

#ifndef KJOEB_CLUSTERING_H
#define KJOEB_CLUSTERING_H

#include "utils.h"

namespace sorting {

    template <class T>
    bool clustersBySize(const Cluster<T> &c1, const Cluster<T> &c2);

}

namespace clustering {

    void rectsByArea(const Cluster <ExtendedRect> &rects, SuperCluster<ExtendedRect> &clusters);

    void rectsByDistance(Cluster<ExtendedRect> &rects, SuperCluster<Line> &clusters, double threshold = 0.9, int maxDistance = std::numeric_limits<int>::max(), int minDistance = 0);

    void linesByAngle(const Cluster<Line>& lines, SuperCluster<Line> &clusters);
}

namespace selecting {
    double linesIntersectRatio(const Cluster<Line>& cluster);
    int linesByIntersectRatio(const SuperCluster<Line>& clusters);

    double rectsVarianceScore(const Cluster<ExtendedRect>& cluster);
    int rectsByVarianceScore(const SuperCluster<ExtendedRect>& clusters, int minSize, int maxSize);
}


#endif //KJOEB_CLUSTERING_H
