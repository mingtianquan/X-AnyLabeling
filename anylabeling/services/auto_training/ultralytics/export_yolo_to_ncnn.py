import os
import subprocess
import shutil
import sys
import torch
from ultralytics import YOLO
import re
import glob
import argparse
from pathlib import Path
from typing import List, Optional, Tuple

def run_command(command: str):
    """doc"""
    print(f": {command}")
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f": {e}")
        raise

class YOLONCNNConverter:
    def __init__(self, model_path: str, model_type: str, model_version: str, imgsz: Optional[int] = None):
        self.model_path = Path(model_path).resolve()
        self.model_type = model_type  # 'det', 'seg', 'pose', 'obb'
        self.model_version = model_version # 'yolov8', 'yolo11'
        self.imgsz = imgsz
        self.model_dir = self.model_path.parent
        self.model_name = self.model_path.stem
        self.output_name = self.model_name.replace("-", "_")
        
        # ?
        self.torchscript_path = self.model_dir / f"{self.model_name}.torchscript"
        self.pnnx_py_path = self.model_dir / f"{self.model_name}_pnnx.py"
        self.output_pt_file = self.model_dir / f"{self.output_name}_new.pt"
        self.final_param_file = self.model_dir / f"{self.output_name}.ncnn.param"
        self.final_bin_file = self.model_dir / f"{self.output_name}.ncnn.bin"

    def _resolve_imgsz(self) -> int:
        """Resolve export/inference size with optional user override."""
        if self.imgsz is not None:
            if self.imgsz <= 0:
                raise ValueError(f"imgsz must be > 0, got {self.imgsz}")
            return self.imgsz
        if self.model_type == 'obb':
            return 1024
        if self.model_type == 'cls':
            return 224
        return 640

    def export_to_torchscript(self):
        """doc"""
        print(f"--- ?  TorchScript  ({self.model_path}) ---")
        model = YOLO(str(self.model_path))
        if self.model_version == 'yolo26':
            head = model.model.model[-1]
            if hasattr(head, "end2end"):
                head.end2end = False
        export_kwargs = dict(format="torchscript", dynamic=False, nms=False)
        if self.imgsz is not None:
            export_kwargs["imgsz"] = self.imgsz
        model.export(**export_kwargs)

    def run_initial_pnnx(self):
        """doc"""
        print("---" " ?  PNNX  ---")
        _pnnx_path = str(self.torchscript_path).replace('\\', '/')
        run_command(f"pnnx {_pnnx_path}")

    def fix_4d_reshape(self, match):
        """doc"""
        op = match.group(1)   # view ?reshape
        current_pos = match.start()
        # ?
        prefix = match.string[:current_pos]
        lines_before = prefix.split('\n')
        
        # 
        current_line = lines_before[-1]
        try:
            # v_106 = v_105.view(1, 128, 20, 20)
            in_var = current_line.split('=')[1].split('.')[0].strip()
        except:
            in_var = "x"

        c1 = match.group(2)
        c2 = match.group(3)
        c3 = match.group(4)
        
        #  A:  view(1, 128, 32, 32) ->  4D 
        if c2.isdigit() and c3.isdigit() and int(c2) in [10, 20, 32, 40, 64, 80, 160]:
            spatial_ref = "x"
            #  view(..., -1) ?(?v_95)
            for i in range(len(lines_before)-1, -1, -1):
                line = lines_before[i]
                #  v_96 = v_95.view(..., -1) 
                flatten_match = re.search(r'(v_\d+) = (v_\d+)\.(view|reshape)\(.*? -1\)', line)
                if flatten_match:
                    spatial_ref = flatten_match.group(2) #  v_95
                    break
                
            #  self ?
            if spatial_ref == "x":
                for i in range(len(lines_before)-1, -1, -1):
                    line = lines_before[i]
                    if ' = self.' in line and '(' in line:
                        m = re.search(r'(v_\d+) = ', line)
                        if m: spatial_ref = m.group(1); break
                        
            return f"{op}(1, {c1}, {spatial_ref}.size(2), {spatial_ref}.size(3))"
        
        #  B: ?view(1, heads, dim, pixels) -> view(1, heads, dim, -1)
        if c2.isdigit() and c3.isdigit() and int(c3) >= 100:
            return f"{op}(1, {c1}, {c2}, -1)"
        
        #  C: view(1, heads, pixels, dim) -> view(1, heads, -1, dim)
        if c2.isdigit() and c3.isdigit() and int(c2) >= 100:
            return f"{op}(1, {c1}, -1, {c3})"
            
        return match.group(0)

    def _remove_unreachable_code(self, content: str) -> str:
        """doc"""
        lines = content.split('\n')
        new_lines = []
        in_forward = False
        forward_returned = False
        
        for line in lines:
            if 'def forward(self,' in line:
                in_forward = True
                forward_returned = False
            
            if in_forward:
                if not forward_returned:
                    new_lines.append(line)
                    if re.search(r'^\s+return\b', line):
                        forward_returned = True
                else:
                    # ?forward 
                    if line.startswith('def ') or line.startswith('class '):
                        in_forward = False
                        new_lines.append(line)
            else:
                new_lines.append(line)
        
        return '\n'.join(new_lines)

    def modify_pnnx_script(self):
        """doc"""
        print("--- ?  PNNX  ---")
        
        #  PNNX ?.py 
        if not self.pnnx_py_path.exists():
            preferred_candidates = [
                self.model_dir / f"{self.output_name}_pnnx.py",
                self.model_dir / f"{self.model_name}_pnnx.py",
                self.model_dir / "model_pnnx.py",
            ]
            generic_candidates = sorted(
                self.model_dir.glob("*_pnnx.py"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            seen = set()
            for tmp_path in preferred_candidates + generic_candidates:
                key = str(tmp_path.resolve()) if tmp_path.exists() else str(tmp_path)
                if key in seen:
                    continue
                seen.add(key)
                if tmp_path.exists():
                    self.pnnx_py_path = tmp_path
                    break
        
        if not self.pnnx_py_path.exists():
            raise FileNotFoundError(f" PNNX ?Python : {self.pnnx_py_path}")

        print(f": {self.pnnx_py_path}")
        with open(self.pnnx_py_path, "r", encoding="utf-8") as f:
            content = f.read()

        if self.model_version == 'yolov8':
            # YOLOv8: ?()
            content = re.sub(r'(view|reshape)\(1, (\d+), (\d{3,})\)', r'\1(1, \2, -1).transpose(1, 2)', content)
            #  cat ?(dim=2 -> dim=1)
            content = re.sub(r'torch\.cat\((.*?), dim=2\)', r'torch.cat(\1, dim=1)', content)
            content = re.sub(r'torch\.cat\((.*?), dim=-1\)', r'torch.cat(\1, dim=1)', content)
        elif self.model_version == 'yolo11':
            # YOLO11: keep previous behavior.
            content = re.sub(r'(view|reshape)\(1, (\d+), (\d{3,})\)', r'\1(1, \2, -1).transpose(1, 2)', content)
            #  4D 
            content = re.sub(r'(view|reshape)\(1, (\d+), (\d+), (\d+)\)', self.fix_4d_reshape, content)
            #  cat ?(?3-tensor )
            content = re.sub(r'(torch\.cat\(\(v_\d+, v_\d+, v_\d+\), dim=)2\)', r'\g<1>1)', content)
            content = re.sub(r'(torch\.cat\(\(v_\d+, v_\d+, v_\d+\), dim=)-1\)', r'\g<1>1)', content)
        elif self.model_version == 'yolo26':
            # YOLO26 decoded heads rely on [1, C, N] for internal anchor decode math.
            # Only make N dynamic; do not transpose to [1, N, C].
            content = re.sub(r'(view|reshape)\(1, (\d+), (\d{3,})\)', r'\1(1, \2, -1)', content)
            content = re.sub(r'(view|reshape)\(1, (\d+), (\d+), (\d+)\)', self.fix_4d_reshape, content)

        # 
        save_pattern = r'mod\.save\([\'\"]([^\'\"]+)[\'\"]\)'
        new_pt_path = str(self.output_pt_file).replace('\\', '/') # ?
        content = re.sub(save_pattern, f'mod.save(r\"{new_pt_path}\")', content)

        #  return 
        content = self._update_return_statement(content)
        
        if self.model_version == 'yolo11' or self.model_version == 'yolo26':
            #  forward ?return ?PNNX 
            content = self._remove_unreachable_code(content)

        with open(self.pnnx_py_path, "w", encoding="utf-8") as f:
            f.write(content)

    def _update_return_statement(self, content: str) -> str:
        """doc"""
        def extract_return_vars(text: str):
            m = re.search(r'^\s*return\s+(v_\d+)\s*$', text, flags=re.M)
            if m:
                ret_var = m.group(1)
                m_assign = re.search(rf'^\s*{re.escape(ret_var)}\s*=\s*\(([^)]*)\)\s*$', text, flags=re.M)
                if m_assign:
                    vars_in_tuple = re.findall(r'v_\d+', m_assign.group(1))
                    if vars_in_tuple:
                        return vars_in_tuple

            m = re.search(r'^\s*return\s*\(([^)]*)\)\s*$', text, flags=re.M)
            if m:
                vars_in_tuple = re.findall(r'v_\d+', m.group(1))
                if vars_in_tuple:
                    return vars_in_tuple

            m = re.search(r'^\s*return\s+([^\n]+)$', text, flags=re.M)
            if m:
                vars_in_tuple = re.findall(r'v_\d+', m.group(1))
                if vars_in_tuple:
                    return vars_in_tuple

            return []

        if self.model_version == 'yolov8':
            triple_cat_pattern = r'(v_\d+) = torch\.cat\(\(v_\d+, v_\d+, v_\d+\), dim=1\)'
        elif self.model_version == 'yolo11' or self.model_version == 'yolo26':
            triple_cat_pattern = r'(v_\d+) = torch\.cat\(\(v_\d+, v_\d+, v_\d+\), dim=\s*1\)'
        else:
            return content # Should not happen

        triple_cats = re.findall(triple_cat_pattern, content)

        if self.model_type == 'det':
            if self.model_version == 'yolo26':
                # yolo26-det first-pass pnnx output is already compatible with C++ decode path.
                # Avoid brittle second-pass return rewrites.
                return content
            if triple_cats:
                if len(triple_cats) >= 2:
                    # Detect newer exports may split bbox and cls into separate 3-way cats.
                    # Keep C++ decode compatibility by merging to [1, N, 64+nc].
                    v_bbox = triple_cats[-2]
                    v_cls = triple_cats[-1]
                    ret_stmt = f"return torch.cat(({v_bbox}, {v_cls}), dim=2)"
                    if ret_stmt not in content:
                        line_pattern = rf'({re.escape(v_cls)} = torch\.cat\(\(v_\d+, v_\d+, v_\d+\), dim=\s*1\))'
                        content = re.sub(line_pattern, f'\\1\\n        {ret_stmt}', content)
                else:
                    v_final = triple_cats[-1]
                    ret_stmt = f"return {v_final}"
                    if ret_stmt not in content:
                        line_pattern = rf'({re.escape(v_final)} = torch\.cat\(\(v_\d+, v_\d+, v_\d+\), dim=\s*1\))'
                        content = re.sub(line_pattern, f'\\1\\n        {ret_stmt}', content)
        
        elif self.model_type == 'seg':
            if self.model_version == 'yolo26':
                # yolo26-seg first-pass pnnx output is already compatible with C++ fallback
                # (merged det+mask, proto). Avoid brittle second-pass return rewrites.
                return content

            return_vars = extract_return_vars(content)
            proto_var = return_vars[-1] if return_vars else None

            if len(triple_cats) >= 2:
                ret_stmt = None
                if self.model_version == 'yolov8':
                    if len(triple_cats) >= 3:
                        # yolov8 Segment: bbox(64), cls(nc), mask(32) are split cats.
                        # C++ expects out0=det(64+nc), out1=mask coeff, out2=proto.
                        if not proto_var:
                            return content
                        v_bbox = triple_cats[-3]
                        v_cls = triple_cats[-2]
                        v_mask = triple_cats[-1]
                        ret_stmt = f"return torch.cat(({v_bbox}, {v_cls}), dim=2), {v_mask}, {proto_var}"
                    else:
                        if not proto_var:
                            return content
                        v_194 = triple_cats[-1]
                        v_157 = triple_cats[-2]
                        ret_stmt = f"return {v_194}, {v_157}, {proto_var}"
                elif self.model_version == 'yolo11':
                    if len(triple_cats) >= 3:
                        # yolo11 Segment newer exports split bbox/cls/mask into three cats.
                        # C++ expects out0=det(64+nc), out1=mask coeff, out2=proto.
                        if not proto_var:
                            return content
                        v_bbox = triple_cats[-3]
                        v_cls = triple_cats[-2]
                        v_mask = triple_cats[-1]
                        ret_stmt = f"return torch.cat(({v_bbox}, {v_cls}), dim=2), {v_mask}, {proto_var}"
                    else:
                        if not proto_var:
                            return content
                        v_final_cat = triple_cats[-1]
                        v_proto_cat = triple_cats[-2]
                        ret_stmt = f"return {v_final_cat}, {v_proto_cat}, {proto_var}"
                
                if ret_stmt and ret_stmt not in content:
                    inserted = False
                    if proto_var:
                        # Preferred path: duplicate proto branch right after mask cat and return early.
                        # This avoids executing rewritten decode branch that may become shape-incompatible.
                        anchor_pattern = rf'^\s*{re.escape(triple_cats[-1])}\s*=\s*torch\.cat\(\(v_\d+, v_\d+, v_\d+\), dim=\s*1\)\s*$'
                        anchor_match = re.search(anchor_pattern, content, flags=re.M)
                        proto_assign_pattern = rf'^\s*{re.escape(proto_var)}\s*=\s*[^\n]+$'
                        proto_assign_match = re.search(proto_assign_pattern, content, flags=re.M)

                        if anchor_match and proto_assign_match and proto_assign_match.start() > anchor_match.end():
                            # Rebuild the proto assignment chain backwards from proto_var.
                            assign_pattern = r'^\s*(v_\d+)\s*=\s*([^\n]+)$'
                            assign_matches = list(re.finditer(assign_pattern, content, flags=re.M))
                            assign_map = {m.group(1): m for m in assign_matches}
                            chain = []
                            seen_vars = set()
                            cur_var = proto_var
                            anchor_pos = anchor_match.end()

                            while cur_var in assign_map and cur_var not in seen_vars:
                                m_cur = assign_map[cur_var]
                                if m_cur.start() <= anchor_pos:
                                    break
                                chain.append(m_cur)
                                seen_vars.add(cur_var)
                                deps = re.findall(r'v_\d+', m_cur.group(2))
                                next_var = None
                                for dep in deps:
                                    m_dep = assign_map.get(dep)
                                    if m_dep and anchor_pos < m_dep.start() < m_cur.start():
                                        next_var = dep
                                        break
                                if not next_var:
                                    break
                                cur_var = next_var

                            if chain:
                                chain.reverse()
                                proto_block = "\n".join(m.group(0) for m in chain)
                                insertion = f"\n{proto_block}\n        {ret_stmt}"
                                content = content[:anchor_match.end()] + insertion + content[anchor_match.end():]
                                inserted = True

                        if not inserted and proto_assign_match:
                            # Fallback: insert return after original proto assignment.
                            content = re.sub(
                                proto_assign_pattern,
                                f'\\g<0>\\n        {ret_stmt}',
                                content,
                                count=1,
                                flags=re.M
                            )
                            inserted = True
                    if not inserted:
                        # Safety fallback: keep original script unchanged if insertion anchor is not found.
                        return content

        elif self.model_type == 'pose' or self.model_type == 'obb':
            if len(triple_cats) >= 2:
                if len(triple_cats) >= 3:
                    # Pose/OBB exports split bbox/cls/extra into three cats.
                    # C++ expects out0=det(64+nc), out1=extra(kpts/angle).
                    v_bbox = triple_cats[-3]
                    v_cls = triple_cats[-2]
                    v_extra = triple_cats[-1]
                    ret_stmt = f"return torch.cat(({v_bbox}, {v_cls}), dim=2), {v_extra}"
                else:
                    v_final = triple_cats[-1]
                    v_extra = triple_cats[-2]
                    ret_stmt = f"return {v_final}, {v_extra}"
                if ret_stmt not in content:
                    line_pattern = rf'({re.escape(triple_cats[-1])} = torch\.cat\(\(v_\d+, v_\d+, v_\d+\), dim=\s*1\))'
                    content = re.sub(line_pattern, f'\\1\\n        {ret_stmt}', content)
        
        return content

    def generate_new_torchscript(self):
        """doc"""
        print("--- ?  TorchScript ---")
        sys.path.append(str(self.model_dir))
        
        namespace = {"torch": torch, "__file__": str(self.pnnx_py_path)}
        with open(self.pnnx_py_path, "r", encoding="utf-8") as f:
            code = f.read()
            
        try:
            exec(code, namespace)
            if "export_torchscript" in namespace:
                namespace["export_torchscript"]()
            else:
                print("?export_torchscript?..")
                model = namespace["Model"]()
                model.eval()
                # BB ?1024
                size = self._resolve_imgsz()
                example_input = torch.rand(1, 3, size, size)
                traced = torch.jit.trace(model, example_input)
                traced.save(str(self.output_pt_file))
            print(f": {self.output_pt_file}")
        except Exception as e:
            print(f": {e}")
            raise

    def run_final_pnnx(self):
        """doc"""
        print("--- ? ?PNNX  ---")
        processed_output_pt_file = str(self.output_pt_file).replace('\\', '/')
        if self.imgsz is None:
            # When imgsz is not provided, keep model/export default tracing shape.
            cmd = f"pnnx {processed_output_pt_file}"
        else:
            size = self._resolve_imgsz()
            half_size = size // 2
            cmd = f"pnnx {processed_output_pt_file} inputshape=[1,3,{size},{size}] inputshape2=[1,3,{half_size},{half_size}]"
        try:
            run_command(cmd)
        except Exception:
            pnnx_param = self.model_dir / f"{self.output_pt_file.stem}.ncnn.param"
            pnnx_bin = self.model_dir / f"{self.output_pt_file.stem}.ncnn.bin"
            if pnnx_param.exists() and pnnx_bin.exists():
                print("! pnnx returned non-zero, but ncnn files were generated. Continue.")
            else:
                raise

    def _remove_legacy_meta_file(self):
        """Do not keep sidecar .meta files for output artifacts."""
        legacy_meta = self.model_dir / f"{self.output_name}.meta"
        if legacy_meta.exists():
            try:
                legacy_meta.unlink()
                print(f"  ? {legacy_meta.name}")
            except Exception as e:
                print(f"  ! failed to remove {legacy_meta.name}: {e}")

    def finalize_files(self):
        """doc"""
        print("--- ?  ---")
        # PNNX ?[.ncnn.param
        #  .stem ?.pt ?
        pnnx_param = self.model_dir / f"{self.output_pt_file.stem}.ncnn.param"
        pnnx_bin = self.model_dir / f"{self.output_pt_file.stem}.ncnn.bin"

        if pnnx_param.exists() and pnnx_bin.exists():
            shutil.copy2(pnnx_param, self.final_param_file)
            shutil.copy2(pnnx_bin, self.final_bin_file)
            self._remove_legacy_meta_file()
            print(f"! ?\n  {self.final_param_file}\n  {self.final_bin_file}")
        else:
            print("Error: failed to find generated ncnn files from pnnx.")
            # 
            params = list(self.model_dir.glob("*.ncnn.param"))
            if params:
                print(f"Found candidate param file: {params[0].name}")
                shutil.copy2(params[0], self.final_param_file)
                shutil.copy2(list(self.model_dir.glob("*.ncnn.bin"))[0], self.final_bin_file)
                self._remove_legacy_meta_file()

    def finalize_initial_pnnx_files(self):
        """Use first-pass pnnx ncnn files directly."""
        print("--- ? (use initial pnnx ncnn) ---")
        init_param = self.model_dir / f"{self.output_name}.ncnn.param"
        init_bin = self.model_dir / f"{self.output_name}.ncnn.bin"
        if not init_param.exists() or not init_bin.exists():
            raise FileNotFoundError(f"Initial ncnn files not found: {init_param} / {init_bin}")

        if init_param.resolve() != self.final_param_file.resolve():
            shutil.copy2(init_param, self.final_param_file)
        if init_bin.resolve() != self.final_bin_file.resolve():
            shutil.copy2(init_bin, self.final_bin_file)
        self._remove_legacy_meta_file()

        print(f"! ?\n  {self.final_param_file}\n  {self.final_bin_file}")

    def cleanup(self):
        """doc"""
        print("---  ---")
        current_script = Path(__file__).resolve()
        
        temp_files_to_delete = [\
            self.torchscript_path,\
            self.pnnx_py_path,\
            self.output_pt_file,\
            # These are common intermediate files
            self.model_dir / f"{self.output_pt_file.stem}.ncnn.param",\
            self.model_dir / f"{self.output_pt_file.stem}.ncnn.bin",\
            self.model_dir / f"{self.output_pt_file.stem}.pnnx.param",\
            self.model_dir / f"{self.output_pt_file.stem}.pnnx.bin",\
            self.model_dir / f"{self.output_pt_file.stem}.pnnx.onnx",\
            self.model_dir / f"{self.output_pt_file.stem}_pnnx.py",\
            self.model_dir / f"{self.output_pt_file.stem}_ncnn.py",\
        ]
        
        # 
        for ext in ['*.torchscript', '*_new.pt', '*_pnnx.py', '*_ncnn.py', '*.pnnx.onnx', '*.pnnx.param', '*.pnnx.bin']:
            for f in self.model_dir.glob(ext):
                # 
                if f.resolve() == current_script:
                    continue
                # 
                if self.model_name in f.name or self.output_name in f.name:
                    if f not in temp_files_to_delete:
                        temp_files_to_delete.append(f)

        # 
        final_files_to_keep = []
        if self.model_type == 'cls':
            # un_initial_pnnx  .ncnn 
            final_files_to_keep.append(self.model_dir / f"{self.model_name}.ncnn.param")
            final_files_to_keep.append(self.model_dir / f"{self.model_name}.ncnn.bin")
        else:
            #  finalize_files ?
            final_files_to_keep.append(self.final_param_file)
            final_files_to_keep.append(self.final_bin_file)

        for f in temp_files_to_delete:
            if f.exists():
                # ?
                if f.resolve() == current_script or f in final_files_to_keep:
                    continue
                try:
                    f.unlink()
                    print(f"  ? {f.name}")
                except Exception as e:
                    print(f"   {f.name}: {e}")

    def convert(self, auto_cleanup=True):
        """doc"""
        try:
            self.export_to_torchscript()
            self.run_initial_pnnx()
            
            if self.model_type != 'cls':
                # yolo26 heads are more sensitive to second-pass torchscript rewrite.
                # Use first-pass pnnx ncnn files directly to avoid shape-mismatch regressions.
                if self.model_version == 'yolo26':
                    self.finalize_initial_pnnx_files()
                else:
                    # Use dynamic-shape conversion path for det/seg/pose/obb across versions.
                    self.modify_pnnx_script()
                    self.generate_new_torchscript()
                    self.run_final_pnnx()
                    self.finalize_files()
            
            if auto_cleanup:
                self.cleanup()
                
            print(f"\n[SUCCESS] {self.model_name} ({self.model_version} {self.model_type}) converted.")
        except Exception as e:
            print(f"\n[FAILED] : {e}")

def main():
    parser = argparse.ArgumentParser(description='YOLO to NCNN converter')
    parser.add_argument(
        'model_version',
        choices=['yolo8', 'yolov8', 'yolo11', 'yolov11', 'yolo26', 'yolov26'],
        help='YOLO version alias (e.g., yolo8/yolov8, yolo11/yolov11, yolo26/yolov26)'
    )
    parser.add_argument('model_type', choices=['det', 'seg', 'pose', 'obb', 'cls'], help=' (e.g., det, seg, pose, obb, cls)')
    parser.add_argument('model_path', type=str, help='YOLO .pt ')
    parser.add_argument('--imgsz', type=int, default=None, help='Export/inference size override. Default follows task type.')
    parser.add_argument('--keep-temp', action='store_true', help=' ()')

    args = parser.parse_args()
    
    version_alias = {
        'yolo8': 'yolov8',
        'yolov8': 'yolov8',
        'yolo11': 'yolo11',
        'yolov11': 'yolo11',
        'yolo26': 'yolo26',
        'yolov26': 'yolo26',
    }
    normalized_version = version_alias[args.model_version]

    converter = YOLONCNNConverter(args.model_path, args.model_type, normalized_version, imgsz=args.imgsz)
    converter.convert(auto_cleanup=not args.keep_temp)

if __name__ == "__main__":
    main()



