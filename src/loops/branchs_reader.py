import math
from itertools import combinations
from tabulate import tabulate


class BranchsHandler:

    def __init__(self, objects):
        self.objects = objects

    def get_obj_count(self, obj):
        return self.objects[obj]["count"] if obj in self.objects else []

    def get_obj_repr(self, obj):
        return self.objects[obj]["representations"] if obj in self.objects else []

    def get_obj_kinematics(self, obj):
        return self.objects[obj]["kinematics"] if obj in self.objects else []

    def get_obj_instances(self, obj):
        if obj in self.objects:
            count = self.get_obj_count(obj)
            reprs = self.get_obj_repr(obj)
            if count == 1 and reprs == [obj]:
                return [obj]
            else:
                return [f"{r}{i}" for r in reprs for i in range(count)]
        else:
            return []

    def get_obj_branch_names(self, obj):
        instances = self.get_obj_instances(obj)
        kinematics = self.get_obj_kinematics(obj)
        return [f"{k}_{inst}" for k in kinematics for inst in instances]
    

    def get_obj_number_branch_names(self):
        """
        Return counts of all physical objects, pre quality selections, pre event selection,
        and post event selection, regardless of the representations required.
        """
        physics_objects = ["Lepton", "Muon", "Electron", "FatJet", "Jet", "BJet", "TauJet"]
        branch_names = []
        for sel in ["nPreQS", "nPreES", "nPostES"]: # nPreES is just nPostQS
            branch_names.extend([f"{sel}_{pObj}" for pObj in physics_objects])
        return branch_names


    ## N-body kinematics
    ## ===============================


    def get_nbody_combinations(self, N, different_types_only = True, combo_objects=("Lepton", "MET")):
        # Build all possible particles as (name, obj_type, slot_index)
        particles = []
        for obj in combo_objects:
            reprs = self.get_obj_repr(obj)
            count = self.get_obj_count(obj)
            
            # For MET (or any singleton with no index) we treat slot as a dummy value (e.g., -1)
            if count == 1 and reprs == [obj]:
                for r in reprs:
                    particles.append((r, obj, 0))  # slot 0 but uniqueness rule will handle
            else:
                for slot in range(count):
                    for r in reprs:
                        particles.append((f"{r}{slot}", obj, slot))
        
        result = []
        for combo in combinations(particles, N):
            names, types, slots = zip(*combo)
            # Rule 1: no two particles from same (object type, slot index)
            if len(set(zip(types, slots))) != N:
                continue
            # Rule 2 (optional): all object types must be different
            if different_types_only and len(set(types)) != N:
                continue
            result.append(names)
        
        return result


    def get_nbody_kinematics(self, N, return_basic = False, include_trivial = False):
        kinematics = list(self.multiObject_basic_kinematics)
        if N < 2: return []

        if return_basic:
            return kinematics

        if N == 2:
            kinematics.extend(self.multiObject_2body_kinematics)

        if include_trivial:
            kinematics.extend(self.multiObject_trival_kinematics)   

        if N == 2 and self.multiObjects_include_mt2:     
            kinematics.append("MT2")   

        return kinematics


    def get_nbody_branch_names(self, N, include_trivial = False, different_types_only = True, combo_objects=("Lepton", "MET")):
        """
        Return a list of branch names for N‑body combinations.
        Example: for N=2, combo=('Lepton0','Lepton1'), kinematic='M' -> 'M_Lepton0_Lepton1'
        """
        combos = self.get_nbody_combinations(N, different_types_only, combo_objects)
        kinematics = self.get_nbody_kinematics(N, return_basic = False, include_trivial = include_trivial)
        branch_names = []
        for combo in combos:
            combo_str = "_".join(combo)   
            for kin in kinematics:
                branch_names.append(f"{kin}_{combo_str}")
        return branch_names



    ## Main Methods
    ## ===============================

    def get_int_branch_names(self):
        return self.get_obj_number_branch_names()

    def get_float_branch_names(self):
        branch_names = []

        # Single object branches
        for obj in self.objects:
            branch_names.extend(self.get_obj_branch_names(obj))

        # N-body branches
        for N in range(2, self.multiObjects_Nmax + 1, 1):
            branch_names.extend(self.get_nbody_branch_names(
                N, 
                include_trivial = self.multiObjects_include_trival_kinematics, 
                different_types_only = self.multiObjects_include_same_represenations == False, 
                combo_objects= self.multiObjects_combo_objects_set
            ))

        # Others
        branch_names.extend(global_scalars)
        branch_names.extend(event_shapes)
        branch_names.append("gen_weight") # generator-level weight
        branch_names.append("weight") # analysis weight (to include NLO norm, Lumi ..)

        return branch_names



    def print_branch_names(self, branches, nCols=4):
        nRows = math.ceil(len(branches) / nCols)

        # Build an empty grid
        grid = [[''] * nCols for _ in range(nRows)]

        # Fill column by column, top to bottom
        idx = 1
        for col in range(nCols):
            for row in range(nRows):
                if idx > len(branches):
                    break
                grid[row][col] = f"{idx}: {branches[idx-1]}"
                idx += 1

        # Print as a pretty grid
        print(tabulate(grid, headers = [f"C{c+1}" for c in range(nCols)], tablefmt="default"))


    def print_summary(self):
        int_branch_names = self.get_int_branch_names()
        float_branch_names = self.get_float_branch_names()
        print(f"Defined {len(int_branch_names)} integer branches for physics.")    
        print(f"Defined {len(float_branch_names)} float branches for physics.")
        print(f"Below we show the defined branch_names:\n")
        self.print_branch_names(int_branch_names + float_branch_names, nCols=3)


        Ntot_branches = len(int_branch_names) + len(float_branch_names)
        msg = f"\nGoing to fill {Ntot_branches} data branches."
        if Ntot_branches > 100:
            msg += " This might take a while ! (Around 15 mins per 1M events)"
            print(msg)
            print(" > Note that whether these branch will be filled with data or not entirely depends on your samples.")
            print(" > In many cases, large amount of branches will contain NaN everywhere.")
            print(" > Cleansing of large amounts of NaN entries in tabular data could lead to loss of valuable data if not done carefully.")


if __name__ == '__main__':
    reader = BranchsHandler()
    reader.print_summary()