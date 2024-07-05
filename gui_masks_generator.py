import subprocess
import os
import datetime


def masks(input_dir, output_dir):

    # input_dir = str(input_dir)
    # output_dir = str(args.output_dir)
    print(input_dir)
    if not os.path.exists(input_dir):
        raise("Enter a proper input directory which actually exists! :-( ")

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    imgs = os.listdir(input_dir)
    #print(imgs)
    for img in imgs:
        check = os.path.splitext(os.path.join(output_dir, img))[0]
        #print(check)
        if os.path.exists(check):
            print(f"folder {check} - already created")
            continue


        if img.endswith(".png"):
            inp_path = os.path.join(input_dir, img)
            print(inp_path)
            
            
            ps_command = f"""
                python "scripts/amg.py" \
                --checkpoint "sam_vit_h_4b8939.pth" \
                --model-type "vit_h" \
                --input {inp_path} \
                --output {output_dir} \
                --points-per-batch 3 \
                --pred-iou-thresh 0.92 \
                --stability-score-thresh 0.96 \
                """.strip()
                ### torch-gpu --device "cuda" (default) else "cpu"            

            stdout_log_file = 'stdout_logs.log'
            error_log_file = 'error_logs.log'

            if os.path.exists(error_log_file):
                with open(error_log_file, 'r') as A, open('error_history.log', 'a') as B:
                    text = A.read()
                    B.write(str(text))

            with open(stdout_log_file, 'a') as C, open(error_log_file, 'a') as D:

                result = subprocess.run(ps_command, shell=True, stdout=C, stderr=D, text=True)
            
            stamp = datetime.datetime.now()
            if result.returncode == 0:
                print(f"SAM mask created successfully for {img}")
            else:
                print(f"{stamp} || Error encountered, check {error_log_file}")

            # testing out for only 1 image include break else remove
            # break
            
            