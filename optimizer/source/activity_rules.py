
class ActivityRules:
    """
    Holds the "arena" rules about actions, including prerequisites, post-conditions,
    valid end activities, and XOR branching logic.
    """

    def __init__(self, prerequisites, post_conditions, transition_dict, xor_rules=None):
        self.prerequisites = prerequisites
        self.post_conditions = post_conditions
        self.valid_end_activities = self._find_valid_end_activities(transition_dict)
        self.xor_rules = xor_rules or {}  # default to empty dict if none provided

    def _find_valid_end_activities(self, transition_dict):
        valid_end_activities = set()
        for prefix, agent_dicts in transition_dict.items():
            for agent_transitions in agent_dicts.values():
                if 'zzz_end' in agent_transitions:
                    if prefix:  # make sure prefix is not empty
                        valid_end_activities.add(prefix[-1])  # last activity in prefix
        
        return valid_end_activities


    def is_case_end(self, activity, outstanding_obligations):
        return activity in self.valid_end_activities and not outstanding_obligations

    # def get_available_activities(self, performed):
    #     return [
    #         a for a in self.prerequisites
    #         if all(p in performed for p in self.prerequisites.get(a, []))
    #     ]
    def get_available_activities(self, performed, xor_windows, allowed_activity_number):
        """
        Returns a list of activities that can currently be executed,
        based on prerequisites and XOR group constraints.

        Parameters:
        - performed: List of activities already performed by the case.
        - xor_windows: Dict[anchor_activity] → List[int | None]
            Tracks XOR decisions per anchor. The latest value is either:
            - None → no group selected yet
            - int → index of the locked group for the current XOR decision window

        Returns:
        - List of activity names that are executable in the current state.
        """
        available = []

        for activity in self.prerequisites:
            # 1. Standard prerequisite check
            if not all(p in performed for p in self.prerequisites.get(activity, [])):
                continue

            # 2. Max allowed activity check
            if allowed_activity_number[activity] <=0:
                continue
        

            # 3 2. XOR validation (if activity is governed by XOR rules)
            xor_valid = True
            for anchor, groups in self.xor_rules.items():
                for group_index, group in enumerate(groups):
                    if activity in group:
                        current_window = xor_windows.get(anchor, [])

                        # 2a. If no decision made yet → allow
                        if not current_window or current_window[-1] is None:
                            xor_valid = True
                        # 2b. If group matches current decision → allow
                        elif current_window[-1] == group_index:
                            xor_valid = True
                        # 2c. If group doesn't match current decision → block
                        else:
                            xor_valid = False

                        break  # activity belongs to only one group under one anchor

                if not xor_valid:
                    break  # short-circuit if blocked by any XOR rule

            if xor_valid:
                available.append(activity)

        return available


    # def is_xor_valid(self, activity: str, case) -> bool:
    #     """
    #     Checks if the activity can be executed based on XOR group logic.
    #     An activity is allowed if:
    #     - It’s not part of any XOR group, or
    #     - It’s the first activity executed from its group for the given anchor, or
    #     - It belongs to the same group as an already chosen activity for the anchor.

    #     XOR decisions are stored in case.xor_decisions as:
    #         { anchor_activity: group_id (int) }
    #     """
    #     for anchor, groups in self.xor_rules.items():
    #         # flatten the xor groups into (index, group) pairs
    #         for group_index, group in enumerate(groups):
    #             if activity in group:
    #                 chosen = case.xor_decisions.get(anchor)
    #                 if chosen is None:
    #                     return True  # no group taken yet
    #                 if chosen == group_index:
    #                     return True  # same group already taken
    #                 return False  # different group already taken
    #     return True  # activity not part of any XOR group

    # def is_xor_valid(self, activity: str, case) -> bool:
    #     """
    #     Dynamic XOR validation that supports repeated anchors.
    #     Logic:
    #     - For a given anchor (e.g., 'Check form completeness'), determine if multiple XOR groups exist.
    #     - If one group has been partially or fully executed, block others.
    #     - If the anchor is re-executed, allow all paths to be reconsidered.
    #     """
    #     for anchor, groups in self.xor_rules.items():
    #         if activity not in [act for g in groups for act in g]:
    #             continue  # Skip if activity not part of any group

    #         # Step 1: check if anchor has been performed more than once
    #         anchor_count = case.performed.count(anchor)
    #         if anchor_count > 1:
    #             # print(f"🔁 Anchor '{anchor}' repeated, XOR paths reset")
    #             continue  # Reset XOR block — allow activity

    #         # Step 2: check which group(s) are already partially executed
    #         executed_groups = set()
    #         for group_index, group in enumerate(groups):
    #             if any(act in case.performed for act in group):
    #                 executed_groups.add(group_index)

    #         # Step 3: determine current group
    #         current_group = None
    #         for group_index, group in enumerate(groups):
    #             if activity in group:
    #                 current_group = group_index
    #                 break

    #         # Step 4: block activity if it’s in a different group
    #         if len(executed_groups) > 0 and current_group not in executed_groups:
    #             conflicting_groups = [groups[i] for i in executed_groups if i != current_group]
    #             # print(f"⛔ Activity '{activity}' is in XOR group {current_group}, but other group(s) already started: {conflicting_groups}")
    #             return False

    #     return True  # ✅ Activity is valid if not blocked by XOR logic

    # def is_xor_valid(self, activity: str, case) -> bool:
    #     """
    #     Tracks XOR group use per anchor execution.
    #     Each time the anchor is executed, a new XOR group may be selected.
    #     Activity is allowed if it matches the most recent XOR group, or can establish one.
    #     """
    #     for anchor, groups in self.xor_rules.items():
    #         # Skip anchors that don't govern this activity
    #         if activity not in [act for g in groups for act in g]:
    #             continue

    #         windows = case.xor_windows.get(anchor, [])
    #         if not windows:
    #             # No decision window created — maybe anchor was never hit (shouldn't happen, but fail-safe)
    #             return False  # Strictly block unless a window is open

    #         current_window = windows[-1]  # Most recent decision

    #         # Identify the group for this activity
    #         group_index = None
    #         for i, group in enumerate(groups):
    #             if activity in group:
    #                 group_index = i
    #                 break

    #         if group_index is None:
    #             return True  # Shouldn't happen — not in any group

    #         if current_window is None:
    #             # First activity in this XOR window: lock the group
    #             case.xor_windows[anchor][-1] = group_index
    #             print(f"📌 XOR decision made: {anchor} → group {group_index}")
    #             return True

    #         if group_index == current_window:
    #             return True  # Matches locked group — allowed

    #         # Otherwise: trying to enter a different XOR group — block it
    #         print(f"⛔ Activity '{activity}' blocked by XOR: group {group_index} != active group {current_window}")
    #         return False

    #     return True  # Activity not governed by any XOR rule — allow


    
    def is_activity_allowed(self, activity: str, case) -> bool:
        """
        Determines whether the given activity is executable at this point in the case.
        Enforces that:
        - Valid end activities are only allowed if there are no outstanding obligations.
        - XOR rules are respected.
        """

        # 🚫 Block premature end activities
        if activity in self.valid_end_activities and case.outstanding_obligations:
            # print(f"⛔ Cannot execute end activity '{activity}' due to obligations: {case.outstanding_obligations}")
            return False

        # # 🚫 Block XOR-invalid paths
        # if not self.is_xor_valid(activity, case):
        #     return False
        
        return True

