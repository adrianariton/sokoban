import os
import pickle as pk


class Logger:
    def __init__(self, output_file=None, output_folder="logs/"):
        self.output_file = output_file
        self.output_folder = output_folder

    @property
    def get_file_path(self):
        return f"{self.output_folder}{self.output_file}.log"

    @property
    def get_folder_path(self):
        return f"{self.output_folder}"

    def __call__(self, *args, return_function=False, **kw_args):
        print("log!")

        def print_(*args, **kw_args):
            if self.output_file is None:
                print(*args, **kw_args)
            else:
                os.makedirs(self.get_folder_path, exist_ok=True)
                dir_path = "".join(self.get_file_path.split("/")[:-1])
                os.makedirs(dir_path, exist_ok=True)

                with open(self.get_file_path, "w+") as f:
                    print(*args, **kw_args, file=f)

        if return_function:
            return print_
        else:
            print_(*args, **kw_args)

    def pickle(self, object, file_name=None):
        print("PICKLE!")
        if file_name is None:
            assert self.output_file is not None, "please pickle in a file"
            file_name = self.output_file
        os.makedirs(self.get_folder_path, exist_ok=True)
        dir_path = "/".join(f"{self.output_folder}{file_name}.pkl".split("/")[:-1])
        print(f"{dir_path=}")
        os.makedirs(dir_path, exist_ok=True)
        with open(f"{self.output_folder}{file_name}.pkl", "wb+") as f:
            pk.dump(object, file=f)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False  # Re-raises exception if one occurred
