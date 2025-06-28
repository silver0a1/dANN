# main.tf

resource "vultr_instance" "gpu_dann_app" {
  # Choose a Vultr region that offers GPU instances.
  # Example: "ewr" (New Jersey), "sjc" (Silicon Valley), "fra" (Frankfurt)
  region = "ewr"

  # IMPORTANT: Replace with a Vultr GPU plan ID that suits your needs.
  # Example: "vc2-1c-2gb-amd-gpu" is a basic GPU plan.
  # For more powerful NVIDIA GPUs (A100, A40, etc.), find their specific plan IDs.
  plan = "vc2-1c-2gb-amd-gpu"

  # IMPORTANT: Replace with the actual OS ID for a Vultr "CUDA-ready" Ubuntu LTS image.
  # A standard Ubuntu 22.04 LTS ID is 195, but a CUDA-ready image will have a different ID.
  # You can find available OS IDs using `vultr-cli os list`.
  os_id = 195 # Placeholder: Replace with the actual CUDA-ready Ubuntu LTS OS ID

  # IMPORTANT: Replace with the ID of an SSH key you've uploaded to Vultr.
  # You can find your SSH key IDs using `vultr-cli ssh-key list`.
  ssh_key_ids = ["YOUR_SSH_KEY_ID_HERE"]

  label    = "dann-cupy-gpu-instance"
  hostname = "dann-cupy-gpu"

  # User data script to set up and run the application on instance boot.
  user_data = <<-EOT
    #!/bin/bash
    # Update package lists and install necessary tools
    apt update -y
    apt install -y git python3-venv

    # Clone the dANN repository
    # The script will be located at /root/dANN_repo/dann_xor_python_cupy.py
    git clone https://github.com/silver0a1/dANN.git /root/dANN_repo

    # Navigate into the cloned repository directory
    cd /root/dANN_repo

    # Create and activate a Python virtual environment
    python3 -m venv dann_env
    source dann_env/bin/activate

    # Install CuPy.
    # Assuming the "CUDA-ready" image has a compatible CUDA Toolkit installed,
    # `pip install cupy` should automatically detect it and install the correct version.
    # If you know the exact CUDA version (e.g., 11.x), you could use `pip install cupy-cuda11x`
    # for more specificity.
    pip install cupy

    # Run the dann_xor_python_cupy.py application in the background.
    # `nohup` ensures the process continues even if the SSH session disconnects.
    # Output is redirected to a log file for later inspection.
    nohup python dann_xor_python_cupy.py > /var/log/dann_cupy_app.log 2>&1 &

    # Deactivate the virtual environment (optional, as nohup detaches the process)
    deactivate
  EOT
}

# Output the public IP address of the created instance
output "instance_ip" {
  description = "The public IP address of the GPU instance."
  value       = vultr_instance.gpu_dann_app.main_ip
}
