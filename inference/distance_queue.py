"""Distance queue grouping promoted from Ishan's PR #8 notebook (cell 7c).

Kept detector-neutral: callers supply boxes and seated probabilities. The
notebook is a training/reference artifact and is never executed at runtime.
"""

from __future__ import annotations

import math


def foot_point(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, y2)


def box_height(bbox):
    return max(bbox[3] - bbox[1], 1)


def pair_distance(det_a, det_b, normalize=True):
    ax, ay = foot_point(det_a["bbox"])
    bx, by = foot_point(det_b["bbox"])
    d = math.hypot(ax - bx, ay - by)
    if normalize:
        d = d / ((box_height(det_a["bbox"]) + box_height(det_b["bbox"])) / 2)
    return d


def point_in_polygon(pt, polygon):
    """Ray casting. polygon = [(x, y), ...] in original-image pixel coordinates."""
    x, y = pt
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            x_cross = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < x_cross:
                inside = not inside
    return inside


def group_elongation(points):
    """
    ~1.0 for a round blob (people clustered in a circle), large for a line.
    Ratio of the spread along the group's main axis vs the perpendicular axis.
    """
    n = len(points)
    mx = sum(p[0] for p in points) / n
    my = sum(p[1] for p in points) / n
    a = sum((p[0] - mx) ** 2 for p in points) / n
    c = sum((p[1] - my) ** 2 for p in points) / n
    b = sum((p[0] - mx) * (p[1] - my) for p in points) / n
    half = (a + c) / 2
    root = math.sqrt(((a - c) / 2) ** 2 + b * b)
    l1, l2 = (half + root, half - root)
    if l1 <= 1e-09:
        return 1.0
    return math.sqrt(l1 / max(l2, 1e-09))


def assign_queue_by_distance(
    detections,
    dist_threshold=1.0,
    min_group_size=3,
    normalize=True,
    min_elongation=None,
    queue_zone=None,
    zone_min_fraction=0.5,
    min_group_size_in_zone=2,
    min_classifier_fraction=None,
    min_member_conf=None,
    rescue_max_p_seated=None,
    rescue_head_tol=0.2,
    rescue_allow_chaining=True,
    rescue_min_neighbors=1,
):
    """
    Step 1: link STANDING people whose feet are within dist_threshold (chained into groups).
    Step 2: keep a group as a queue only if it passes the filters:

      NO ZONE given:
        - at least min_group_size people (3 stops two friends chatting counting as a queue)
        - if min_elongation is set: the group must be line-shaped, not a round huddle
      ZONE given (polygon around the serving/queue area):
        - at least zone_min_fraction of the group must stand inside the zone
        - at least min_group_size_in_zone people (2 is fine here, a pair at the counter
          IS a queue). The shape check is skipped, a crowd at the counter still counts.
      EITHER way, if min_classifier_fraction is set: that fraction of the group must also
        have det["queue_vote"] == True (your trained queue classifier agreeing).

    min_member_conf: detections below this confidence never join a group (they still count in
      the headcount). Stops a shaky box from extending a line.
    rescue_max_p_seated: when a queue is found, someone the seated classifier called "seated" but
      with p_seated <= this value is moved INTO the queue if they stand in line with queue members
      (within dist_threshold AND head height within rescue_head_tol of that member's). This fixes
      queue members whose legs are hidden by tables: head+torso looks seated, but their position in
      the line says otherwise. None = off.
    rescue_allow_chaining=False restricts evidence to the original standing queue
    members, preventing rescued seated people from extending a queue recursively.
    rescue_min_neighbors requires this many eligible neighbors in the SAME queue.

    Seated people are otherwise ignored. Sets det["in_queue"], det["queue_group"],
    det["queue_reject"], det["rescued"] in place. Returns (number_of_queue_groups, rejected).
    Rejected is a list of
    {"size": n, "reason": ...} for groups of 2+ people that were linked but NOT called a queue.
    """
    if rescue_min_neighbors < 1:
        raise ValueError("rescue_min_neighbors must be at least 1")
    standing_idx = [
        i
        for i, d in enumerate(detections)
        if not d.get("seated", False)
        and (min_member_conf is None or d.get("confidence", 1.0) >= min_member_conf)
    ]
    parent = {i: i for i in standing_idx}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a in range(len(standing_idx)):
        for b in range(a + 1, len(standing_idx)):
            i, j = (standing_idx[a], standing_idx[b])
            if pair_distance(detections[i], detections[j], normalize) <= dist_threshold:
                parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in standing_idx:
        groups.setdefault(find(i), []).append(i)
    for d in detections:
        d["in_queue"] = False
        d["queue_group"] = None
        d["queue_reject"] = None
        d["rescued"] = False
    n_groups, rejected = (0, [])
    for members in groups.values():
        size = len(members)
        pts = [foot_point(detections[i]["bbox"]) for i in members]
        reason = None
        if queue_zone is not None:
            frac_inside = sum(point_in_polygon(p, queue_zone) for p in pts) / size
            if frac_inside < zone_min_fraction:
                reason = "outside_zone"
            elif size < min_group_size_in_zone:
                reason = "too_small"
        elif size < min_group_size:
            reason = "too_small"
        elif min_elongation is not None and size >= 3 and (group_elongation(pts) < min_elongation):
            reason = "blob"
        if reason is None and min_classifier_fraction is not None:
            votes = sum(1 for i in members if detections[i].get("queue_vote", False))
            if votes / size < min_classifier_fraction:
                reason = "classifier"
        if reason is None:
            for i in members:
                detections[i]["in_queue"] = True
                detections[i]["queue_group"] = n_groups
            n_groups += 1
        else:
            for i in members:
                detections[i]["queue_reject"] = reason
            if size >= 2:
                rejected.append({"size": size, "reason": reason})
    if rescue_max_p_seated is not None:
        original_members = {i for i, d in enumerate(detections) if d["in_queue"]}
        cands = [
            i
            for i, d in enumerate(detections)
            if d.get("seated", False)
            and (not d["in_queue"])
            and (d.get("p_seated") is not None)
            and (d["p_seated"] <= rescue_max_p_seated)
            and (min_member_conf is None or d.get("confidence", 1.0) >= min_member_conf)
        ]
        changed = True
        while changed and cands:
            changed = False
            for i in list(cands):
                c = detections[i]
                if queue_zone is not None and (
                    not point_in_polygon(foot_point(c["bbox"]), queue_zone)
                ):
                    continue
                neighbor_groups: dict[int, list[int]] = {}
                for j, m in enumerate(detections):
                    if j == i or not m["in_queue"]:
                        continue
                    if not rescue_allow_chaining and j not in original_members:
                        continue
                    if pair_distance(c, m, normalize) > dist_threshold:
                        continue
                    h = max(box_height(c["bbox"]), box_height(m["bbox"]))
                    if abs(c["bbox"][1] - m["bbox"][1]) <= rescue_head_tol * h:
                        neighbors = neighbor_groups.setdefault(m["queue_group"], [])
                        neighbors.append(j)
                        if len(neighbors) < rescue_min_neighbors:
                            continue
                        c["in_queue"], c["queue_group"], c["rescued"] = (
                            True,
                            m["queue_group"],
                            True,
                        )
                        cands.remove(i)
                        changed = True
                        break
    return (n_groups, rejected)
