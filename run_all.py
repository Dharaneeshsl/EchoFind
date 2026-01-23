"""
Master script to run the complete pipeline:
1. Train encoder
2. Build retrieval index
3. Evaluate linear probe
4. Generate visualizations
"""
import os
import sys
import subprocess

def run_command(cmd, description):
    """Run a command and handle errors."""
    print("\n" + "=" * 60)
    print(description)
    print("=" * 60)
    try:
        result = subprocess.run(cmd, shell=True, check=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed with error: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠ {description} interrupted by user")
        return False

def main():
    """Run complete pipeline."""
    print("=" * 60)
    print("EchoFind - Complete Pipeline")
    print("=" * 60)
    
    steps = [
        ("python train.py", "Training encoder"),
        ("python build_index.py", "Building retrieval index"),
        ("python evaluate.py", "Evaluating linear probe"),
        ("python visualize.py", "Generating visualizations")
    ]
    
    # Ask user which steps to run
    print("\nAvailable steps:")
    for i, (_, desc) in enumerate(steps, 1):
        print(f"  {i}. {desc}")
    print("  5. Run all steps")
    print("  0. Exit")
    
    choice = input("\nSelect step(s) to run (comma-separated, e.g., 1,2,3 or 5): ").strip()
    
    if choice == "0":
        print("Exiting...")
        return
    
    if choice == "5":
        # Run all steps
        for cmd, desc in steps:
            if not run_command(cmd, desc):
                print(f"\n⚠ Pipeline stopped at: {desc}")
                print("You can continue manually by running the remaining steps.")
                return
        print("\n" + "=" * 60)
        print("✓ All steps completed successfully!")
        print("=" * 60)
    else:
        # Run selected steps
        try:
            indices = [int(x.strip()) - 1 for x in choice.split(",")]
            for idx in indices:
                if 0 <= idx < len(steps):
                    cmd, desc = steps[idx]
                    if not run_command(cmd, desc):
                        print(f"\n⚠ Pipeline stopped at: {desc}")
                        return
        except ValueError:
            print("Invalid input. Please enter comma-separated numbers or 5 for all steps.")
            return

if __name__ == "__main__":
    main()
