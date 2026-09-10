import math
import sys
import argparse
from itertools import combinations
from tabulate import tabulate
from yaml import safe_load 


class BranchesHandler:
    """
    Handles physics object branch name generation and validation
    based on a YAML configuration file.

    Usage:
        reader = BranchesHandler("branch_config.yml")
        reader.print_summary()
        reader.print_config()       # detailed inspection
        reader.is_valid()           # returns True/False
    """

    # Default physics objects for integer branch counts
    _PHYSICS_OBJECTS = ["Lepton", "Muon", "Electron", "FatJet", "Jet", "BJet", "TauJet"]
    _SELECTION_STAGES = ["nPreQS", "nPreES", "nPostES"]

    def __init__(self, config_path=None):
        self.config_path = config_path
        self.objects = {}
        self.global_scalars = []
        self.event_shapes = []

        # Multi-object configuration (defaults)
        self.multiObjects_Nmax = 2
        self.multiObjects_combo_objects_set = ("Lepton", "MET")
        self.multiObjects_include_same_represenations = False
        self.multiObjects_include_trival_kinematics = False
        self.multiObjects_include_mt2 = False
        self.multiObject_basic_kinematics = []
        self.multiObject_2body_kinematics = []
        self.multiObject_trival_kinematics = []

        # Validation results
        self._errors = []
        self._warnings = []

        if config_path:
            self._load_config(config_path)
            self._validate_config()

    # ==============================
    # Config Loading
    # ==============================

    def _load_config(self, config_path):
        """Load and parse the YAML configuration file."""
        try:
            with open(config_path, "r") as f:
                config = safe_load(f)
        except FileNotFoundError:
            self._errors.append(f"Configuration file not found: {config_path}")
            return

        if config is None:
            self._errors.append("Configuration file is empty.")
            return

        # --- Parse objects ---
        self.objects = config.get("objects", {})

        # --- Parse event-variables ---
        ev_vars = config.get("event-variables", {})
        self.global_scalars = ev_vars.get("global_scalars", [])
        self.event_shapes = ev_vars.get("event_shapes", [])

        # --- Parse multi-objects ---
        mo = config.get("multi-objects", {})
        self.multiObjects_Nmax = mo.get("Nmax", 2)
        self.multiObjects_combo_objects_set = tuple(mo.get("combo_set", ("Lepton", "MET")))
        self.multiObjects_include_same_represenations = mo.get("include_same_represenations", False)
        self.multiObjects_include_trival_kinematics = mo.get("include_trival_kinematics", False)
        self.multiObjects_include_mt2 = mo.get("include_mt2", False)
        self.multiObject_basic_kinematics = mo.get("basic_kinematics", [])
        self.multiObject_2body_kinematics = mo.get("2body_kinematics", [])
        self.multiObject_trival_kinematics = mo.get("trival_kinematics", [])

    # ==============================
    # Validation
    # ==============================

    def _validate_config(self):
        """Validate the loaded configuration and populate errors/warnings."""
        self._errors = []
        self._warnings = []

        # --- Validate objects ---
        if not self.objects:
            self._errors.append("No objects defined in configuration.")
        else:
            for obj_name, obj_def in self.objects.items():
                if not isinstance(obj_def, dict):
                    self._errors.append(
                        f"Object '{obj_name}' must be a dictionary with "
                        "'count', 'representations', and 'kinematics'."
                    )
                    continue

                # Check count
                if "count" not in obj_def:
                    self._errors.append(f"Object '{obj_name}' is missing 'count'.")
                elif not isinstance(obj_def["count"], int) or obj_def["count"] < 0:
                    self._errors.append(
                        f"Object '{obj_name}' count must be a non-negative integer, "
                        f"got {obj_def['count']!r}."
                    )

                # Check representations
                if "representations" not in obj_def:
                    self._errors.append(f"Object '{obj_name}' is missing 'representations'.")
                elif not isinstance(obj_def["representations"], list) or len(obj_def["representations"]) == 0:
                    self._errors.append(
                        f"Object '{obj_name}' representations must be a non-empty list."
                    )
                elif not all(isinstance(r, str) for r in obj_def["representations"]):
                    self._errors.append(
                        f"Object '{obj_name}' representations must all be strings. "
                        f"Check for unquoted values in YAML."
                    )

                # Check kinematics
                if "kinematics" not in obj_def:
                    self._errors.append(f"Object '{obj_name}' is missing 'kinematics'.")
                elif not isinstance(obj_def["kinematics"], list) or len(obj_def["kinematics"]) == 0:
                    self._errors.append(
                        f"Object '{obj_name}' kinematics must be a non-empty list."
                    )
                elif not all(isinstance(k, str) for k in obj_def["kinematics"]):
                    self._errors.append(
                        f"Object '{obj_name}' kinematics must all be strings. "
                        f"Check for unquoted values in YAML."
                    )

                # Warn about extra keys
                known_keys = {"count", "representations", "kinematics"}
                extra_keys = set(obj_def.keys()) - known_keys
                if extra_keys:
                    self._warnings.append(
                        f"Object '{obj_name}' has unknown keys: {extra_keys}. These will be ignored."
                    )

        # --- Validate combo_set ---
        if not self.multiObjects_combo_objects_set:
            self._warnings.append("combo_set is empty; no N-body branches will be generated.")
        for obj in self.multiObjects_combo_objects_set:
            if obj not in self.objects:
                self._errors.append(
                    f"Object '{obj}' in combo_set is not defined in 'objects'."
                )

        # --- Validate Nmax ---
        if not isinstance(self.multiObjects_Nmax, int) or self.multiObjects_Nmax < 2:
            self._errors.append(
                f"Nmax must be an integer >= 2, got {self.multiObjects_Nmax!r}."
            )
        else:
            # Check feasibility
            total_particles = self._count_combo_particles()
            if self.multiObjects_Nmax > total_particles:
                self._warnings.append(
                    f"Nmax={self.multiObjects_Nmax} exceeds total available particles "
                    f"({total_particles}) in combo_set. Some N-body combinations will be empty."
                )

        # --- Validate global_scalars and event_shapes ---
        if not isinstance(self.global_scalars, list):
            self._errors.append("global_scalars must be a list.")
        if not isinstance(self.event_shapes, list):
            self._errors.append("event_shapes must be a list.")

        # --- Validate kinematics lists ---
        for name, val in [
            ("basic_kinematics", self.multiObject_basic_kinematics),
            ("2body_kinematics", self.multiObject_2body_kinematics),
            ("trival_kinematics", self.multiObject_trival_kinematics),
        ]:
            if not isinstance(val, list):
                self._errors.append(f"{name} must be a list, got {type(val).__name__}.")
            elif not all(isinstance(k, str) for k in val):
                self._errors.append(f"{name} must contain only strings. Check YAML for unquoted values.")

        # --- Cross-checks ---
        if self.multiObjects_include_mt2 and "MET" not in self.multiObjects_combo_objects_set:
            self._warnings.append(
                "include_mt2 is True but 'MET' is not in combo_set. "
                "MT2 requires MET (Px, Py) and may not be meaningful."
            )

        if self.multiObjects_include_trival_kinematics and not self.multiObject_trival_kinematics:
            self._warnings.append(
                "include_trival_kinematics is True but trival_kinematics list is empty."
            )

    def _count_combo_particles(self):
        """Count total available particles from combo_set objects."""
        total = 0
        for obj in self.multiObjects_combo_objects_set:
            if obj not in self.objects:
                continue
            count = self.objects[obj].get("count", 0)
            reprs = self.objects[obj].get("representations", [])
            if count == 1 and reprs == [obj]:
                total += 1
            else:
                total += count * len(reprs)
        return total

    def is_valid(self):
        """Return True if no validation errors were found."""
        return len(self._errors) == 0

    def print_validation(self):
        """Print validation errors and warnings."""
        if self._errors:
            print("\n" + "=" * 50)
            print("VALIDATION ERRORS")
            print("=" * 50)
            for e in self._errors:
                print(f"  ✗ [ERROR] {e}")

        if self._warnings:
            print("\n" + "=" * 50)
            print("VALIDATION WARNINGS")
            print("=" * 50)
            for w in self._warnings:
                print(f"  ⚠ [WARN]  {w}")

        if not self._errors and not self._warnings:
            print("\n✓ Configuration is valid. No errors or warnings.")

    # ==============================
    # Object Accessors
    # ==============================

    def get_obj_count(self, obj):
        return self.objects[obj]["count"] if obj in self.objects else 0

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
        branch_names = []
        for sel in self._SELECTION_STAGES:
            branch_names.extend([f"{sel}_{pObj}" for pObj in self._PHYSICS_OBJECTS])
        return branch_names

    # ==============================
    # N-body Kinematics
    # ==============================

    def get_nbody_combinations(self, N):
        # Build all possible particles as (name, obj_type, slot_index)
        particles = []
        for obj in self.multiObjects_combo_objects_set:
            reprs = self.get_obj_repr(obj)
            count = self.get_obj_count(obj)

            if count == 0 or not reprs:
                continue

            # For MET (or any singleton with no index) we treat slot as 0
            if count == 1 and reprs == [obj]:
                for r in reprs:
                    particles.append((r, obj, 0))
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
            if self.multiObjects_include_same_representations == False and len(set(types)) != N:
                continue
            result.append(names)

        return result

    def get_nbody_kinematics(self, N, return_basic=False):
        kinematics = list(self.multiObject_basic_kinematics)
        if N < 2:
            return []

        if return_basic:
            return kinematics

        if N == 2:
            kinematics.extend(self.multiObject_2body_kinematics)

        if self.multiObjects_include_trival_kinematics:
            kinematics.extend(self.multiObject_trival_kinematics)

        if N == 2 and self.multiObjects_include_mt2:
            kinematics.append("MT2")

        return kinematics

    def get_nbody_branch_names(self, N):
        """
        Return a list of branch names for N-body combinations.
        Example: for N=2, combo=('Lepton0','Lepton1'), kinematic='M' -> 'M_Lepton0_Lepton1'
        """
        combos = self.get_nbody_combinations(N)
        kinematics = self.get_nbody_kinematics(N)
        branch_names = []
        for combo in combos:
            combo_str = "_".join(combo)
            for kin in kinematics:
                branch_names.append(f"{kin}_{combo_str}")
        return branch_names

    # ==============================
    # Main Methods
    # ==============================

    def get_int_branch_names(self):
        return self.get_obj_number_branch_names()

    def get_float_branch_names(self):
        branch_names = []

        # Single object branches
        for obj in self.objects:
            branch_names.extend(self.get_obj_branch_names(obj))

        # N-body branches
        for N in range(2, self.multiObjects_Nmax + 1, 1):
            branch_names.extend(self.get_nbody_branch_names(N))

        # Others
        branch_names.extend(self.global_scalars)
        branch_names.extend(self.event_shapes)
        branch_names.append("gen_weight")   # generator-level weight
        branch_names.append("weight")       # analysis weight (NLO norm, Lumi, ...)

        return branch_names

    # ==============================
    # Printing / Inspection
    # ==============================

    def print_branch_names(self, branches, nCols=4):
        nRows = math.ceil(len(branches) / nCols)

        # Build an empty grid
        grid = [[""] * nCols for _ in range(nRows)]

        # Fill column by column, top to bottom
        idx = 1
        for col in range(nCols):
            for row in range(nRows):
                if idx > len(branches):
                    break
                grid[row][col] = f"{idx}: {branches[idx - 1]}"
                idx += 1

        print(tabulate(grid, headers=[f"C{c + 1}" for c in range(nCols)], tablefmt="default"))

    def print_config(self):
        """Print the parsed configuration for inspection (triggered by --inspect)."""
        print("=" * 60)
        print("CONFIGURATION INSPECTION")
        print("=" * 60)

        # --- Objects ---
        print("\n--- Objects ---\n")
        obj_data = []
        for name, obj_def in self.objects.items():
            obj_data.append([
                name,
                obj_def.get("count", "N/A"),
                ", ".join(obj_def.get("representations", [])),
                len(obj_def.get("kinematics", [])),
            ])
        print(tabulate(obj_data, headers=["Object", "Count", "Representations", "# Kinematics"],
                       tablefmt="grid"))

        # --- Detailed kinematics per object ---
        print("\n--- Object Kinematics ---\n")
        for name, obj_def in self.objects.items():
            kins = obj_def.get("kinematics", [])
            print(f"  {name} ({len(kins)}): {kins}")

        # --- Event variables ---
        print("\n--- Event Variables ---\n")
        print(f"  global_scalars ({len(self.global_scalars)}): {self.global_scalars}")
        print(f"  event_shapes   ({len(self.event_shapes)}): {self.event_shapes}")

        # --- Multi-object settings ---
        print("\n--- Multi-Object Settings ---\n")
        mo_data = [
            ["Nmax", self.multiObjects_Nmax],
            ["combo_set", list(self.multiObjects_combo_objects_set)],
            ["include_same_represenations", self.multiObjects_include_same_represenations],
            ["include_trival_kinematics", self.multiObjects_include_trival_kinematics],
            ["include_mt2", self.multiObjects_include_mt2],
            ["basic_kinematics", self.multiObject_basic_kinematics],
            ["2body_kinematics", self.multiObject_2body_kinematics],
            ["trival_kinematics", self.multiObject_trival_kinematics],
        ]
        print(tabulate(mo_data, headers=["Setting", "Value"], tablefmt="grid"))

        # --- Available particles ---
        print("\n--- Available Particles for N-body Combinations ---\n")
        particles = []
        for obj in self.multiObjects_combo_objects_set:
            if obj in self.objects:
                count = self.objects[obj].get("count", 0)
                reprs = self.objects[obj].get("representations", [])
                if count == 1 and reprs == [obj]:
                    particles.append(obj)
                else:
                    for slot in range(count):
                        for r in reprs:
                            particles.append(f"{r}{slot}")
        print(f"  Particles ({len(particles)}): {particles}")

        # --- Combination summary ---
        print("\n--- N-body Combination Summary ---\n")
        combo_data = []
        for N in range(2, self.multiObjects_Nmax + 1):
            combos = self.get_nbody_combinations(
                N,
                different_types_only=not self.multiObjects_include_same_represenations,
                combo_objects=self.multiObjects_combo_objects_set,
            )
            all_kins = self.get_nbody_kinematics(
                N, return_basic=False,
                include_trivial=self.multiObjects_include_trival_kinematics,
            )
            n_branches = len(combos) * len(all_kins)
            combo_data.append([N, len(combos), len(all_kins), n_branches])
        print(tabulate(combo_data,
                       headers=["N", "# Combos", "# Kinematics", "# Branches"],
                       tablefmt="grid"))

        # --- Branch count summary ---
        print("\n--- Branch Count Summary ---\n")
        int_branches = self.get_int_branch_names()
        float_branches = self.get_float_branch_names()
        summary_data = [
            ["Integer branches", len(int_branches)],
            ["Float branches", len(float_branches)],
            ["  - Single-object", sum(len(self.get_obj_branch_names(obj)) for obj in self.objects)],
        ]
        for N in range(2, self.multiObjects_Nmax + 1):
            nbody = self.get_nbody_branch_names(
                N,
                include_trivial=self.multiObjects_include_trival_kinematics,
                different_types_only=self.multiObjects_include_same_represenations == False,
                combo_objects=self.multiObjects_combo_objects_set,
            )
            summary_data.append([f"  - {N}-body", len(nbody)])
        summary_data.extend([
            ["  - Global scalars", len(self.global_scalars)],
            ["  - Event shapes", len(self.event_shapes)],
            ["  - Weights", 2],
            ["TOTAL", len(int_branches) + len(float_branches)],
        ])
        print(tabulate(summary_data, headers=["Category", "Count"], tablefmt="grid"))

    def print_summary(self):
        """Print a summary of all defined branches."""
        int_branch_names = self.get_int_branch_names()
        float_branch_names = self.get_float_branch_names()

        print(f"\nDefined {len(int_branch_names)} integer branches for physics.")
        print(f"Defined {len(float_branch_names)} float branches for physics.")
        print(f"\nBelow we show the defined branch_names:\n")
        self.print_branch_names(int_branch_names + float_branch_names, nCols=3)

        Ntot_branches = len(int_branch_names) + len(float_branch_names)
        msg = f"\nGoing to fill {Ntot_branches} data branches."
        if Ntot_branches > 100:
            msg += " This might take a while ! (Around 15 mins per 1M events)"
            print(msg)
            print(" > Note that whether these branches will be filled with data or not")
            print("   entirely depends on your samples.")
            print(" > In many cases, large amounts of branches will contain NaN everywhere.")
            print(" > Cleansing of large amounts of NaN entries in tabular data could lead")
            print("   to loss of valuable data if not done carefully.")


def main():

    parser = argparse.ArgumentParser(
        description="Read, check, and validate branch configuration from a YAML file."
    )
    parser.add_argument(
        "config",
        nargs="?",
        default="branch_config.yml",
        help="Path to the YAML configuration file (default: branch_config.yml)",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Print detailed configuration inspection before the branch summary.",
    )
    args = parser.parse_args()

    reader = BranchesHandler(args.config)

    # Always print validation results
    reader.print_validation()

    if args.inspect:
        reader.print_config()

    reader.print_summary()

    # Exit with error code if validation failed
    if not reader.is_valid():
        print("\n✗ Configuration has errors. Please fix them before proceeding.")
        sys.exit(1)
    else:
        print("\n✓ Configuration validated successfully.")


if __name__ == "__main__":
    main()