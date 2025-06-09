from collections import defaultdict
import pandas as pd
import math
import scipy.stats as st
from datetime import datetime, timedelta

def sample_from_distribution(distribution):
    """
    Copy of the AgentSimulator utils method
    """
    if distribution.type.value == "expon":
        scale = distribution.mean - distribution.min
        if scale < 0.0:
            print("Warning! Trying to generate EXPON sample with 'mean' < 'min', using 'mean' as scale value.")
            scale = distribution.mean
        sample = st.expon.rvs(loc=distribution.min, scale=scale, size=1)
    elif distribution.type.value == "gamma":
        # If the distribution corresponds to a 'gamma' with loc!=0, the estimation is done wrong
        # dunno how to take that into account
        sample = st.gamma.rvs(
            pow(distribution.mean, 2) / distribution.var,
            loc=0,
            scale=distribution.var / distribution.mean,
            size=1,
        )
    elif distribution.type.value == "norm":
        sample = st.norm.rvs(loc=distribution.mean, scale=distribution.std, size=1)
    elif distribution.type.value == "uniform":
        sample = st.uniform.rvs(loc=distribution.min, scale=distribution.max - distribution.min, size=1)
    elif distribution.type.value == "lognorm":
        # If the distribution corresponds to a 'lognorm' with loc!=0, the estimation is done wrong
        # dunno how to take that into account
        pow_mean = pow(distribution.mean, 2)
        phi = math.sqrt(distribution.var + pow_mean)
        mu = math.log(pow_mean / phi)
        sigma = math.sqrt(math.log(phi ** 2 / pow_mean))
        sample = st.lognorm.rvs(sigma, loc=0, scale=math.exp(mu), size=1)
    elif distribution.type.value == "fix":
        sample = [distribution.mean] * 1

    return sample[0]

def remove_transitive_response_constraints(response_constraints):
    cleaned = {}

    for act, responses in response_constraints.items():
        direct = set(responses)

        # Remove any activity that is indirectly reachable through other responses
        for mid in responses:
            indirects = set(response_constraints.get(mid, []))
            direct -= indirects

        cleaned[act] = list(direct)

    return cleaned


def discover_post_conditions(df, activity_col='activity', case_col='case_id', order_by='end_time'):
    response_counts = defaultdict(lambda: defaultdict(int))
    activity_counts = defaultdict(int)

    # Group by case
    for case_id, group in df.groupby(case_col):
        sorted_activities = group.sort_values(by=order_by)[activity_col].tolist()

        for i, act in enumerate(sorted_activities):
            activity_counts[act] += 1

            # All activities that come after this one
            for after_act in sorted_activities[i+1:]:
                response_counts[act][after_act] += 1

    # Build final response constraint map
    post_conditions = {}
    for act, after_acts in response_counts.items():
        constraints = []
        for after_act, count in after_acts.items():
            # Threshold: e.g., B happens after A in 80%+ of A occurrences
            if count / activity_counts[act] >= 0.8:
                constraints.append(after_act)
        if constraints:
            post_conditions[act] = constraints

    post_conditions = remove_transitive_response_constraints(post_conditions)

    return post_conditions


def extract_all_successors(transition_dict):
    """
    Converts a nested transition dictionary to a flat mapping:
    prefix_activity → list of all possible successor activities (non-zero prob)

    Parameters:
    - transition_dict: dict of {prefix: {agent_id: {activity: prob}}} ! Careful, only one activity prefixes are valid

    Returns:
    - dict of {activity: [possible next activities]}
    """
    flat_successors = {}
    seen_anchors = set()

    for prefix, agent_dict in transition_dict.items():
        if not prefix:
            continue  # skip empty prefixes
        if prefix[-1] in seen_anchors:
            continue
        anchor = prefix[-1]  # last activity in the prefix
        seen_anchors.add(anchor)

        successor_set = set()
        for agent_transitions in agent_dict.values():
            for act, prob in agent_transitions.items():
                if prob > 0:
                    successor_set.add(act)

        flat_successors[anchor] = sorted(successor_set)

    return flat_successors


def mine_concurrent_activities(df, case_col='case_id', activity_col='activity',
                                start_col='start_time', end_col='end_time'):
    """
    For each activity, detect other activities that can run concurrently
    by overlapping time windows in the same case.

    Parameters:
    - df: Event log with case_id, activity, start_time, end_time

    Returns:
    - co_occurrence: dict {activity: [other activities that overlapped with it]}
    """
    df = df.copy()
    df[start_col] = pd.to_datetime(df[start_col])
    df[end_col] = pd.to_datetime(df[end_col])

    co_occurrence = defaultdict(set)

    for case_id, group in df.groupby(case_col):
        group = group.sort_values(by=start_col)
        for i, row_i in group.iterrows():
            act_i, start_i, end_i = row_i[activity_col], row_i[start_col], row_i[end_col]
            for j, row_j in group.iterrows():
                if i == j:
                    continue
                act_j, start_j, end_j = row_j[activity_col], row_j[start_col], row_j[end_col]
                # Check for overlap
                if start_i < end_j and start_j < end_i:
                    co_occurrence[act_i].add(act_j)

    # Convert sets to sorted lists
    return {act: sorted(list(others)) for act, others in co_occurrence.items()}


def extract_xor_groups_from_cooccurrence(successor_map, co_occurrence_map):
    """
    Builds XOR groups from a simplified co-occurrence map.
    An activity is excluded from XOR groups if it can co-occur with the anchor.
    Returns:
    - xor_groups: {anchor: list of mutually exclusive groups (each group is a list of activities)}
    """
    xor_groups = defaultdict(list)

    for anchor, successors in successor_map.items():
        if not successors:
            continue

        # ⚠️ Filter out successors that can co-occur with the anchor
        filtered_successors = [
            act for act in successors
            if act not in co_occurrence_map.get(anchor, []) and anchor not in co_occurrence_map.get(act, [])
        ]

        remaining = set(filtered_successors)
        groups = []

        while remaining:
            act = remaining.pop()
            group = {act}

            for other in list(remaining):
                if (
                    act in co_occurrence_map.get(other, []) or
                    other in co_occurrence_map.get(act, [])
                ):
                    group.add(other)
                    remaining.remove(other)

            groups.append(sorted(group))

        if len(groups) > 1:
            xor_groups[anchor] = groups

    return xor_groups

# Note to self. This does not cover cases like: Reject application cannot happen after Approve application

def discover_prerequisites_from_log(df, activity_col='activity', case_col='case_id', order_by='end_time'):
    # Step 1: Collect all activities that appear before each activity in each case
    activity_to_preceding_sets = defaultdict(list)

    for case_id, group in df.groupby(case_col):
        sorted_activities = group.sort_values(by=order_by)[activity_col].tolist()
        seen = set()
        for i, act in enumerate(sorted_activities):
            activity_to_preceding_sets[act].append(seen.copy())
            seen.add(act)

    # Step 2: Intersect the "seen-before" sets across all cases
    raw_prerequisites = {}
    for act, preceding_sets in activity_to_preceding_sets.items():
        if preceding_sets:
            raw_prerequisites[act] = set.intersection(*preceding_sets)
        else:
            raw_prerequisites[act] = set()

    # Step 3: Remove transitive dependencies
    # If A → B and B → C, remove A from prerequisites of C
    def remove_transitive(prereq_dict):
        cleaned = {}
        for act in prereq_dict:
            direct_prereqs = prereq_dict[act].copy()
            # Remove any indirect dependencies
            for p in direct_prereqs.copy():
                indirects = prereq_dict.get(p, set())
                direct_prereqs -= indirects
            cleaned[act] = list(direct_prereqs)
        return cleaned

    strict_prerequisites = remove_transitive(raw_prerequisites)
    return strict_prerequisites



def shift_activity_start_to_next_valid_window(start_time: pd.Timestamp, duration: float, calendar_json: list, max_days = 3,  step = timedelta(days=1)) -> pd.Timestamp:
    """
    Shift the activity start time forward to the beginning of the next available calendar window (shift)
    where the full activity duration fits.

    Parameters:
    - start_time: Proposed start time that violated the current calendar window
    - duration: Activity duration in seconds
    - calendar_json: List of weekly calendar windows from RCalendar.intervals_to_json()
    - max_days: Max days the function will check forward to avoid infinite loops. Defaults to 3 to account for weekends.
    - steps: how much afterwards the function skips to (made to work with days)

    Returns:
    - A new pd.Timestamp at the start of a valid calendar window, or None if no window is found

    Notes:
    - Assumes all calendar windows are same-week recurring (i.e., a 7-day schedule)
    - Made to work with calendars with one shift per day
    - Skips over malformed time strings silently
    """

    for offset in range(1, max_days + 1):
        future_day = start_time + offset * step
        day_name = future_day.strftime('%A').upper()

        for entry in calendar_json:
            if entry['from'] != day_name:
                continue

            try:
                shift_start_time = datetime.strptime(entry['beginTime'], '%H:%M:%S').time()
                shift_end_time = datetime.strptime(entry['endTime'], '%H:%M:%S').time()
            except ValueError as ve:
                print(f"⚠️ Malformed calendar entry: {ve}")
                continue

            proposed_start = datetime.combine(future_day.date(), shift_start_time).replace(tzinfo=start_time.tzinfo)
            proposed_end = proposed_start + timedelta(seconds=duration)

            # Only accept if it fits entirely in the shift window
            if proposed_end.time() <= shift_end_time:
                return pd.Timestamp(proposed_start)

    return None  # No valid window found


def compute_transition_weight(
    performed: list[str],
    last_agent: int,
    transition_probabilities: dict,
    agent_transition_probabilities: dict,
    is_orchestrated: float,
    proposal_agent: int,
    activity: str,
) -> float:
    """
    Weights regarding historic behavior:
    - If the process is autonomous then we only need the agent_transition_probabilities
    - If the process is orchestrated then we need to use transition_probabilities

    Weights regarding optimization:
    - tbd

    Returns:
    - joint_waigt: float
    """
    if not performed:
        return 0.0  # First activity → no context for prediction
    
    if is_orchestrated:
        prefix = tuple(performed)
        # P(activity | prefix, agent)
        agent_sim_weight = transition_probabilities.get(prefix, {}).get(proposal_agent, {})
    else:
        last_activity = performed[-1]
        # P(agent, activity | last_agent, last_activity)
        agent_sim_weight = agent_transition_probabilities.get(last_agent, {}).get(last_activity, {}).get(proposal_agent, {}).get(activity, 0.0)
        # if last_activity == 'Check application form completeness':
        #     print("last_activity 'Check application form completeness")
        #     print(activity, proposal_agent, agent_sim_weight)

    transition_weight = agent_sim_weight
    
    return transition_weight

def validate_simulated_log(df, prerequisites, post_conditions, valid_end_activities, 
                            case_col='case_id', activity_col='activity', order_by='start'):
    issues = []

    for case_id, group in df.groupby(case_col):
        sorted_activities = group.sort_values(by=order_by)[activity_col].tolist()

        if not sorted_activities:
            issues.append((case_id, "Empty trace"))
            continue

        # Remove zzz_end for logic checks
        activities_no_end = [a for a in sorted_activities if a != "zzz_end"]

        # 🚨 1. Prerequisites check
        performed = set()
        for act in activities_no_end:
            required = prerequisites.get(act, [])
            if not all(pre in performed for pre in required):
                missing = [pre for pre in required if pre not in performed]
                issues.append((case_id, f"Activity '{act}' missing prerequisites {missing}"))
            performed.add(act)

        # 🚨 2. Post-conditions check
        for i, act in enumerate(activities_no_end):
            required_posts = post_conditions.get(act, [])
            future_acts = set(activities_no_end[i+1:])
            for post in required_posts:
                if post not in future_acts:
                    issues.append((case_id, f"Activity '{act}' missing required post-condition '{post}'"))

        # 🚨 3. End correctness check
        if activities_no_end:
            last_real_activity = activities_no_end[-1]
            if last_real_activity not in valid_end_activities:
                issues.append((case_id, f"Case ends incorrectly on '{last_real_activity}'"))

    return issues
