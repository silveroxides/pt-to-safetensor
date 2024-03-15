from argparse import ArgumentParser, RawTextHelpFormatter
from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha256
from json import dumps
from os import listdir, path, stat, walk
from pathlib import PurePath
from textwrap import dedent
from time import perf_counter
from typing import Dict, List, Optional, Set, Union
import torch
from safetensors.torch import load_file, save_file

@dataclass
class ProcessArgs:
    convert_path: str
    accel: bool
    skip_meta: bool
    json: bool
    html: bool

def check_file_size(st_filename: str, pt_filename: str):
    st_size = stat(st_filename).st_size
    pt_size = stat(pt_filename).st_size
    if (st_size - pt_size) / pt_size > 0.01:
      raise RuntimeError(f"The file size different is more than 1%: - {st_filename}: {st_size} - {pt_filename}: {pt_size}")

def postconvert_check(pt_model: dict, pt_filename: str, st_filename: str, device: Union[str, torch.device]):
    # Compare the file sizes
    check_file_size(st_filename, pt_filename)

    # CUDA device number is required by load_file()/safe_open() rather than just accepting 'cuda' like torch.load()
    #   See https://github.com/huggingface/safetensors/blob/main/bindings/python/src/lib.rs
    #   Should parse 'cuda' or 'cuda:0' but doesn't, possibly because derive & IntoPy still force Cuda to be an integer?
    load_device = (torch.cuda.current_device() if device == torch.device('cuda') else device)

    # Load the safetensors model and compare tensor size/elements to the pt/bin model
    st_model = load_file(st_filename, device=load_device)
    for k in pt_model:
        pt_tensor = pt_model[k]
        st_tensor = st_model[k]
        if not torch.equal(pt_tensor, st_tensor):
            raise RuntimeError(f"The output tensors do not match for key {k}")
    return f"Conversion successful to {st_filename}"

def shared_pointers(tensors):
    ptrs = defaultdict(list)
    for k, v in tensors.items():
        ptrs[v.data_ptr()].append(k)
    failing = []
    for ptr, names in ptrs.items():
        if len(names) > 1:
            failing.append(names)
    return failing

def convert_file(filename: str, device: Union[str, torch.device], args: ProcessArgs):
    print(f"Converting {filename}...")

    # Generate the SHA256 checksum of the original file
    h = sha256()
    with open(filename, 'rb') as f:
        while chunk := f.read(65536):
            h.update(chunk)
    filesum = h.hexdigest()

    # Load the PyTorch model
    model = torch.load(filename, map_location=device)

    # Get the extension
    file_extension = PurePath(filename).suffix

    # Use state_dict as the model if present
    if "state_dict" in model:
        model = model["state_dict"]

    # Set initial metadata
    st_metadata={"original_filename": PurePath(filename).name, "original_hash": filesum, "original_legacy_hash": filesum[0:8]}

    if file_extension == '.pt':
        #Extract the embedding tensors
        model_tensors = model.get('string_to_param').get('*')
        s_model = { 'emb_params': model_tensors }

        # Store the requested training information, if it exists
        if ('name' in model) and (model['name'] is not None):
            st_metadata['training_name'] = model['name']
        if ('sd_checkpoint_name' in model) and (model['sd_checkpoint_name'] is not None):
            st_metadata['sd_checkpoint_name'] = model['sd_checkpoint_name']
        if ('sd_checkpoint' in model) and (model['sd_checkpoint'] is not None):
            st_metadata['sd_checkpoint_hash'] = model['sd_checkpoint']
        if ('step' in model) and (model['step'] is not None):
            st_metadata['step'] = str(model['step'])

        # Replace the original model with the tensors version for saving
        model = s_model
    elif file_extension == '.bin':
        shared = shared_pointers(model)
        for shared_weights in shared:
            for name in shared_weights[1:]:
                model.pop(name)
        # For tensors to be contiguous
        model = {k: v.contiguous() for k, v in model.items()}
        # Just use the first value from the model dict as the tensors for adding metadata
        # This will not report accurate information for VAEs, it is just a hack for .bin embeddings
        model_tensors = list(model.values())[0]

    # Store and print the tensor's shape (vectors and dimensions) if available
    if isinstance(model_tensors, torch.Tensor):
        tensor_vectors = ''
        if len(list(model_tensors.shape)) == 1:
            tensor_vectors = '1'
        elif len(list(model_tensors.shape)) == 2:
            tensor_vectors = str(model_tensors.shape[0])
        st_metadata['vectors'] = tensor_vectors
        st_metadata['vector_dim'] = str(model_tensors.shape[-1])

    # Sort the metadata by value so it is saved in a consistent order on each conversion
    st_metadata = dict(sorted(st_metadata.items()))

    # Print the metadata contents
    print("---------")
    for key, value in st_metadata.items():
        print("{0:<21}".format(key) + ': ' + value)
    print("---------")

    # Optionally save the metadata to a file
    if args.skip_meta == False:
        if args.json == True:
            meta_filename = str(PurePath(filename).with_suffix('.metadata.json'))
            meta_file = open(meta_filename, "w")
            meta_file.write(dumps(st_metadata, indent=2))
            meta_file.close
        if args.html == True:
            meta_filename = str(PurePath(filename).with_suffix('.metadata.html'))
            meta_file = open(meta_filename, "w")
            print("<!DOCTYPE html>", file=meta_file)
            print("<html lang=en>", file=meta_file)
            print("  <title>" + PurePath(filename).stem + "</title>", file=meta_file)
            print("    <body>\n      <h3>Metadata</h3>", file=meta_file)
            print("      <ul>", file=meta_file)
            for key, value in st_metadata.items():
                print("      <li>" + key + ':&nbsp;' + value + '</li>', file=meta_file)
            print("    </ul>\n  </body>\n</html>", file=meta_file)
            meta_file.close
        else:
            meta_filename = str(PurePath(filename).with_suffix('.metadata.txt'))
            meta_file = open(meta_filename, "w")
            for key, value in st_metadata.items():
                print("{0:<21}".format(key) + ': ' + value, file=meta_file)
            meta_file.close

    # Save the converted model
    st_filename = str(PurePath(filename).with_suffix('.safetensors'))
    save_file(model, st_filename)
    # Validate size and compare tensors - passing the model to avoid reloading the original file
    postconvert_check(model, filename, st_filename, device)

def process_files(convert_path: str, device: Union[str, torch.device], args: ProcessArgs) -> str:
    if path.isfile(convert_path) and (PurePath(convert_path).suffix == '.pt' or PurePath(convert_path).suffix == '.bin'):
        # Path is a .pt or .bin file, process this file
        convert_file(convert_path, device, args)
    elif path.isdir(convert_path):
        # Path is a directory, process all .pt or .bin files in the directory and its subdirectories
        for folder, subs, files in walk(convert_path):
            for filename in files:
                if PurePath(filename).suffix == '.pt' or PurePath(filename).suffix == '.bin':
                    convert_file(path.join(folder, filename), device, args)
                    print()
    else:
        return f"{convert_path} is not a valid directory or .pt/.bin file."

def main():
    # Set up the argparse help and arguments
    DESCRIPTION = dedent('''\
        Simple tool to automatically convert some weights to `safetensors` format.

        *** WARNING: files will be executed when converting - any malicious code will be executed, too. ***
        *** Do not run this on your own machine with untrusted/unscanned files containing pickletensors ***

        If you get an error on a specific file, it may just have the wrong extension, e.g. try renaming .bin to .pt
        Safetensors metadata will be added detailing the original format and shape of the tensor (vectors, dimensions)
        The .pt conversion will display and save metadata about the training model, hash, and steps, when available.
        The .bin conversion does not provide those extra details, the format seems to lack that data.
        Post-conversion, the script will check file sizes and compare that the output tensors match the original.

        Credit to https://github.com/DiffusionDalmation and https://github.com/huggingface/safetensors for original steps
        ''')

    parser = ArgumentParser(description=DESCRIPTION, formatter_class=RawTextHelpFormatter)
    parser.add_argument(
        "convert_path",
        type=str,
        help="The full path of a filename or directory of files to convert",
    )
    parser.add_argument(
        "--accel",
        action='store_true',
        help="Use CUDA (NVIDIA) or Metal (MacOS) instead of CPU",
    )
    parser.add_argument(
        "--skip-meta",
        action='store_true',
        help="Skip creating metadata files",
    )
    parser.add_argument(
        "--json",
        action='store_true',
        help="Write metadata files as JSON",
    )
    parser.add_argument(
        "--html",
        action='store_true',
        help="Write metadata files as HTML",
    )
    # Parse arguments
    args = parser.parse_args()
    process_args = ProcessArgs(**vars(parser.parse_args()))

    device = 'cpu'

    # Try to use Metal or CUDA instead of CPU if --accel was passed in, fall back to CPU
    if args.accel == True:
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Using device \"{device}\" for conversion.")
    print()

    # Start the timer as a rough benchmark and process the path we were given
    print("Starting conversion process...")
    print()
    start_time = perf_counter()

    process_files(args.convert_path, device, args)

    # We're done
    total_time = round(perf_counter() - start_time, 4)
    print()
    print(f"Completed in {total_time} seconds.")

if __name__ == "__main__":
    # Start the process
    main()
