"""Blocks flagged IPs. DRY-RUN by default: only prints commands. Real blocking needs Linux + root + enforce=True."""
import ipaddress, subprocess, platform

def block(ips, enforce=False, whitelist=("127.0.0.1",)):
    results = []
    for ip in ips:
        try:
            a = ipaddress.ip_address(ip)
        except ValueError:
            results.append({"ip": ip, "status": "invalid"}); continue
        if ip in whitelist or a.is_loopback or a.is_multicast:
            results.append({"ip": ip, "status": "whitelisted"}); continue
        cmd = ["iptables", "-I", "INPUT", "-s", ip, "-j", "DROP"]
        if enforce and platform.system() == "Linux":
            r = subprocess.run(cmd, capture_output=True, text=True)
            results.append({"ip": ip, "status": "blocked" if r.returncode == 0 else "error: " + r.stderr.strip()})
        else:
            results.append({"ip": ip, "status": "dry-run", "cmd": " ".join(cmd)})
    return results
