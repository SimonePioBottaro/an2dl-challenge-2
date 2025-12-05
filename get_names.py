import os
import argparse

def get_filenames(directory, output_file):
    """Extract all filenames from a directory and save to a text file."""
    try:
        filenames = os.listdir(directory)
        filenames = [f for f in filenames if os.path.isfile(os.path.join(directory, f))]
        filenames.sort()
        
        with open(output_file, 'w') as f:
            for filename in filenames:
                f.write(filename + '\n')
        
        print(f"✓ Found {len(filenames)} files. Saved to {output_file}")
    except FileNotFoundError:
        print(f"Error: Directory '{directory}' not found.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract filenames from a directory")
    parser.add_argument("directory", help="Path to the directory")
    parser.add_argument("-o", "--output", default="filenames.txt", help="Output file (default: filenames.txt)")
    
    args = parser.parse_args()
    get_filenames(args.directory, args.output)