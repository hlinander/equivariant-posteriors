import argparse
from pathlib import Path

# pyelftools is the library used for parsing ELF files
from elftools.elf.elffile import ELFFile
from elftools.common.exceptions import ELFError
import pefile

# Define the memory page size, same as in the C++ version
PAGE_SIZE = 0x1000


def process_elf_file(file_path: Path):
    """
    Parses a single file to check if it's a 64-bit ELF and processes its
    loadable segments.

    For each LOAD segment, it reads its data and pads it to the next
    full page boundary.

    Args:
        file_path: A Path object pointing to the file.

    Returns:
        True if the file was a valid 64-bit ELF and was processed,
        False otherwise.
    """
    try:
        with open(file_path, "rb") as f:
            elf = ELFFile(f)

            # 1. Filter for 64-bit ELF files only
            if elf.elfclass != 64:
                print(type(elf.elfclass), elf.elfclass)
                return False

            print(f"[+] Processing 64-bit ELF: {file_path}")

            # 2. Iterate over all segments (Program Headers)
            ret = []
            for segment in elf.iter_segments():
                header = segment.header

                # We are only interested in loadable segments
                if header.p_type == "PT_LOAD":
                    original_size = header.p_filesz

                    if original_size == 0:
                        # Skip segments with no data in the file (like .bss)
                        continue

                    # Determine if segment is code or data by checking executable flag
                    # PF_X (execute) flag is 0x1
                    is_executable = header.p_flags & 0x1
                    segment_type = "CODE" if is_executable else "DATA"

                    # Read the entire segment's data from the file
                    segment_data = segment.data()

                    # --- Padding Logic ---
                    # Calculate the total size needed to hold all the data, rounded
                    # up to the nearest page boundary.
                    padded_size = (original_size + PAGE_SIZE - 1) & ~(PAGE_SIZE - 1)

                    # Create the padded data by appending zero bytes
                    padding_needed = padded_size - original_size
                    padded_data = segment_data + (b"\x00" * padding_needed)
                    ret.append((padded_data, segment_type))

                    # --- Output ---
                    # print("    -> Segment Found")
                    # print(f"       Type: {segment_type}")
                    # print(
                    #     f"       Original Size: {original_size} bytes (0x{original_size:x})"
                    # )
                    # print(
                    #     f"       Padded Size: {padded_size} bytes (0x{padded_size:x})"
                    # )

            # print()  # Add a newline for cleaner output
            return ret

    except ELFError:
        # This happens if the file is not an ELF file. We can safely ignore it.
        return []
    except IOError as e:
        # Handle cases where we don't have permission to read the file
        # print(f"Could not read file {file_path}: {e}")
        return []


def process_pe_file(file_path: Path):
    """
    Parses a single file to check if it's a 64-bit PE and processes its sections.
    If successful, it returns a list of tuples containing the padded data and
    type for each section. Otherwise, it returns an empty list.
    """
    try:
        pe = pefile.PE(str(file_path), fast_load=True)

        # 1. Filter for 64-bit PE files only (AMD64 / x86-64)
        if pe.FILE_HEADER.Machine != pefile.MACHINE_TYPE["IMAGE_FILE_MACHINE_AMD64"]:
            pe.close()
            print(f"{pe.FILE_HEADER.Machine:x}")
            return []

        print(f"[+] Processing 64-bit PE: {file_path}")

        # 2. Iterate over all sections and collect results
        ret = []
        for section in pe.sections:
            original_size = section.SizeOfRawData
            if original_size == 0:
                continue

            is_executable = (
                section.Characteristics
                & pefile.SECTION_CHARACTERISTICS["IMAGE_SCN_MEM_EXECUTE"]
            )
            section_type = "CODE" if is_executable else "DATA"
            section_data = section.get_data()

            padded_size = (original_size + PAGE_SIZE - 1) & ~(PAGE_SIZE - 1)
            padding_needed = padded_size - original_size
            padded_data = section_data + (b"\x00" * padding_needed)
            ret.append((padded_data, section_type))

        pe.close()
        return ret

    except pefile.PEFormatError:
        # Not a PE file, return empty list
        return []
    except Exception:
        # Handle other potential errors, like permission denied
        return []


def main():
    """
    Main function to set up argument parsing and start the directory scan.
    """
    parser = argparse.ArgumentParser(
        description="Recursively scan a directory and parse 64-bit ELF files."
    )
    parser.add_argument("directory", help="The directory path to scan.")
    args = parser.parse_args()

    root_path = Path(args.directory)

    if not root_path.is_dir():
        print(f"Error: Provided path '{root_path}' is not a directory.")
        return

    print(f"Starting recursive scan of: {root_path}")
    print("========================================")
    print()

    # Use rglob('*') to recursively find all files and directories
    for entry in root_path.rglob("*"):
        if entry.is_file():
            print(process_pe_file(entry))

    print("========================================")
    print("Scan complete.")


if __name__ == "__main__":
    main()
