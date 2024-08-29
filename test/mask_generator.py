import subprocess
import os
import argparse

parser = argparse.ArgumentParser(
    description= "Code to create binary masks for each image \
        in given directory into another directory"
)

parser.add_argument("--input_dir",
                    type=str,
                    required=True,
                    help="Path to a folder of images.",  )

parser.add_argument("--output_dir",
                    type=str,
                    required=True,
                    help="Path to folder which will contain a separate folder \
                          for masks for each input image",  )

def main(args):
    input_dir = str(args.input_dir)
    output_dir = str(args.output_dir)
    print(input_dir)
    if not os.path.exists(input_dir):
        raise("Enter a proper input directory which actually exists! :-( ")

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    imgs = os.listdir(input_dir)
    print(imgs)
    for img in imgs:
        check = os.path.splitext(os.path.join(output_dir, img))[0]
        if os.path.exists(check):
            print(f"folder {check} - already created")
            continue
        if img.endswith(".png"):
            inp_path = os.path.join(input_dir, img)
            ps_command = f"""
                python "D:/UCC/Thesis/segment-anything-main/scripts/amg.py" \
                --checkpoint "D:/UCC/Thesis/segment-anything-main/sam_vit_h_4b8939.pth" \
                --model-type "vit_h" \
                --input {inp_path} \
                --output {output_dir} \
                --device "cpu"
                """.strip()
             
            stdout_log_file = 'stdout_logs.log'
            error_log_file = 'error_logs.log'

            if os.path.exists(error_log_file):
                with open(error_log_file, 'r') as A, open('error_history.log', 'a') as B:
                    text = A.read()
                    B.write(str(text))

            with open(stdout_log_file, 'a') as C, open(error_log_file, 'a') as D:

                result = subprocess.run(ps_command, shell=True, stdout=C, stderr=D, text=True)

            if result.returncode == 0:
                print("Code executed correctly")
            else:
                print(f"Error encountered, check {error_log_file}")
            

    
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)