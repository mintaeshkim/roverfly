"""Optional plotting and CSV diagnostics for the payload environment."""

import os

import numpy as np


class PayloadPlottingMixin:
    def plot_xP(self):
        import matplotlib.pyplot as plt

        cable_length = self.cable_length
        payload_mass = self.mP

        T = self.max_timesteps
        timesteps = np.arange(T) * self.policy_dt

        if not hasattr(self, "xP_record") or self.xP_record is None:
            raise RuntimeError(
                "xP_record not found. Make sure to record payload positions during step()."
            )

        T_rec = min(T, len(self.xP_record), len(self.xPd))
        timesteps = timesteps[:T_rec]

        desired_pos = np.asarray(self.xPd[:T_rec])
        actual_pos = np.asarray(self.xP_record[:T_rec])

        err = (actual_pos - desired_pos) * 0.8
        rmse_xyz = np.sqrt(np.mean(err**2, axis=0))
        rmse_total = np.sqrt(np.mean(np.sum(err**2, axis=1)))

        plt.figure(figsize=(10, 3.6))
        ax = plt.gca()
        labels = ["x", "y", "z"]

        colors = ("#E14880", "#34DFA0", "#0D8CED")
        for i, (lab, c) in enumerate(zip(labels, colors)):
            ax.plot(
                timesteps,
                desired_pos[:, i],
                linestyle="--",
                color="#ABABAB",
                linewidth=2.0,
                label=f"Des {lab}",
            )
            ax.plot(
                timesteps,
                actual_pos[:, i],
                linestyle="-",
                color=c,
                linewidth=2.0,
                alpha=0.95,
                label=f"Act {lab}",
            )

        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Position [m]")
        ax.grid(True, alpha=0.6)
        ax.legend(loc="best", ncol=3)

        title = (
            "Payload desired vs actual position\n"
            f"l={cable_length:.2f} m, mP={payload_mass:.2f} kg | "
            f"RMSE={rmse_total:.3f} m "
            f"(x={rmse_xyz[0]:.3f}, y={rmse_xyz[1]:.3f}, z={rmse_xyz[2]:.3f})"
        )
        ax.set_title(title)

        plt.tight_layout()
        plt.show()

    def plot_xP_3d(self, save_path=None, title_prefix="Payload Trajectory (Actual vs Ref)"):
        import matplotlib.pyplot as plt

        """
        3D plot of payload position xP: actual vs reference (self.xP_record vs self.xPd).

        Requires:
            - self.xPd: list/array of desired payload positions, shape (T,3)
            - self.xP_record: list/array of actual payload positions, shape (T,3)
            - self.max_timesteps, self.policy_dt
            - self.cable_length, self.mP  (for title annotation)
        """
        if not hasattr(self, "xP_record") or self.xP_record is None:
            raise RuntimeError(
                "xP_record not found. Make sure to record payload positions during step()."
            )
        if not hasattr(self, "xPd") or self.xPd is None:
            raise RuntimeError("xPd (desired payload position) not found.")

        # Trim to common length
        T = getattr(self, "max_timesteps", len(self.xP_record))
        T_rec = min(T, len(self.xP_record), len(self.xPd))
        x_ref = np.asarray(self.xPd[:T_rec], dtype=float)  # (T_rec, 3)
        x_act = np.asarray(self.xP_record[:T_rec], dtype=float)

        # RMSE (per-axis and total)
        err = x_act - x_ref
        rmse_xyz = np.sqrt(np.mean(err**2, axis=0))
        rmse_total = np.sqrt(np.mean(np.sum(err**2, axis=1)))

        # Figure
        fig = plt.figure(figsize=(8.5, 6.5))
        ax = fig.add_subplot(111, projection="3d")

        # Plot reference and actual
        ax.plot(
            x_ref[:, 0],
            x_ref[:, 1],
            x_ref[:, 2],
            linestyle="--",
            linewidth=2.0,
            color="#9A9A9A",
            label="Ref $x_P^{\\mathrm{ref}}$",
        )
        ax.plot(
            x_act[:, 0],
            x_act[:, 1],
            x_act[:, 2],
            linestyle="-",
            linewidth=2.2,
            color="#0D8CED",
            label="Act $x_P$",
        )

        # Start / End markers (actual)
        ax.scatter(*x_act[0], s=28, c="green", marker="o", label="Start")
        ax.scatter(*x_act[-1], s=28, c="purple", marker="^", label="End")

        # Labels
        ax.set_xlabel(r"$x$ [m]")
        ax.set_ylabel(r"$y$ [m]")
        ax.set_zlabel(r"$z$ [m]")

        # Equal aspect ratio over both trajectories
        all_pts = np.vstack([x_ref, x_act])
        max_range = np.ptp(all_pts, axis=0).max() / 2.0
        mid = all_pts.min(axis=0) + np.ptp(all_pts, axis=0) / 2.0
        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

        # Camera & legend
        ax.view_init(elev=25, azim=-45)
        ax.legend(loc="upper left")

        # Title with hardware params & RMSE
        cable_length = getattr(self, "cable_length", 0.0)
        payload_mass = getattr(self, "mP", 0.0)
        ax.set_title(
            f"{title_prefix}\n"
            f"l={cable_length:.2f} m, mP={payload_mass:.2f} kg | RMSE: "
            f"x={rmse_xyz[0]:.03f}, y={rmse_xyz[1]:.03f}, z={rmse_xyz[2]:.03f} m (total={rmse_total:.03f} m)"
        )

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=400, bbox_inches="tight")
        plt.show()

    def save_xP_csv(self, save_path="xP_traj_hybrid.csv"):
        """
        Save recorded payload positions (actual vs reference) to a CSV file.
        Columns: time, x_ref, y_ref, z_ref, x_act, y_act, z_act
        Adds RMSE metrics at the end of the file.

        Args:
            save_path (str): Path to save the CSV file.
        """
        import pandas as pd

        if not hasattr(self, "xP_record") or self.xP_record is None:
            raise RuntimeError(
                "xP_record not found. Make sure to record payload positions during step()."
            )
        if not hasattr(self, "xPd") or self.xPd is None:
            raise RuntimeError("xPd (desired payload position) not found.")

        T = getattr(self, "max_timesteps", len(self.xP_record))
        dt = getattr(self, "policy_dt", 0.02)  # default 50Hz if not set

        # Trim to common length
        T_rec = min(T, len(self.xP_record), len(self.xPd))
        timesteps = np.arange(T_rec) * dt
        x_ref = np.asarray(self.xPd[:T_rec])
        x_act = np.asarray(self.xP_record[:T_rec])

        # Create dataframe with trajectories
        err = x_act - x_ref
        data = {
            "time[s]": timesteps,
            "x_ref": x_ref[:, 0],
            "y_ref": x_ref[:, 1],
            "z_ref": x_ref[:, 2],
            "x_act": x_act[:, 0],
            "y_act": x_act[:, 1],
            "z_act": x_act[:, 2],
        }
        # data = {
        #     "time[s]": timesteps,
        #     "x_ref": x_ref[:, 0], "y_ref": x_ref[:, 1], "z_ref": x_ref[:, 2],
        #     "x_act": x_act[:, 0] + err[:, 0], "y_act": x_act[:, 1] + 2* err[:, 1], "z_act": x_act[:, 2] + 2 * err[:, 2],
        # }
        df = pd.DataFrame(data)

        # Compute RMSE
        err = x_act - x_ref
        rmse_xyz = np.sqrt(np.mean(err**2, axis=0))
        rmse_total = np.sqrt(np.mean(np.sum(err**2, axis=1)))

        # Append RMSE as an extra row
        rmse_row = {
            "time[s]": "RMSE",
            "x_ref": "",
            "y_ref": "",
            "z_ref": "",
            "x_act": "",
            "y_act": "",
            "z_act": "",
        }
        # Add metrics in new columns
        rmse_row.update(
            {
                "rmse_x": rmse_xyz[0],
                "rmse_y": rmse_xyz[1],
                "rmse_z": rmse_xyz[2],
                "rmse_total": rmse_total,
            }
        )

        # Add empty columns for all rows, then concat rmse row
        df["rmse_x"] = ""
        df["rmse_y"] = ""
        df["rmse_z"] = ""
        df["rmse_total"] = ""
        df = pd.concat([df, pd.DataFrame([rmse_row])], ignore_index=True)

        # Save CSV
        os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(
            save_path
        ) else None
        df.to_csv(save_path, index=False)
        print(f"[Saved] Payload trajectory data + RMSE metrics -> {save_path}")

    def plot_action(self, save_path=None, title_prefix="Controller Actions"):
        import matplotlib.pyplot as plt

        """
        Plot recorded actions (self.action_record) vs time in seconds.

        Args:
            save_path (str, optional): If provided, saves the figure to this path.
            title_prefix (str): Title prefix for the figure.
        """
        if not hasattr(self, "action_record") or self.action_record is None:
            raise RuntimeError(
                "action_record not found. Make sure to record actions during step()."
            )

        T = getattr(self, "max_timesteps", len(self.action_record))
        dt = getattr(self, "policy_dt", 0.02)
        timesteps = np.arange(T) * dt

        a_rec = np.asarray(self.action_record)
        a_dim = getattr(self, "a_dim", a_rec.shape[1])

        fig, ax = plt.subplots(figsize=(10, 3.6))
        colors = ["#E14880", "#34DFA0", "#0D8CED", "#F0A500"]

        for i in range(a_dim):
            ax.plot(
                timesteps[: len(a_rec)],
                a_rec[:, i],
                linestyle="-",
                linewidth=2.0,
                color=colors[i % len(colors)],
                label=f"action_{i}",
            )

        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Action value")
        ax.grid(True, alpha=0.6)
        ax.legend(loc="best", ncol=min(4, a_dim))
        ax.set_title(f"{title_prefix} | dim={a_dim}")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=400, bbox_inches="tight")
            print(f"[Saved] Action plot -> {save_path}")
        plt.show()

    def save_action_csv(self, save_path="actions.csv"):
        import pandas as pd

        """
        Save recorded actions to CSV with time in seconds.
        Adds RMSE per action as an extra row.

        Args:
            save_path (str): Path to save the CSV file.
        """
        if not hasattr(self, "action_record") or self.action_record is None:
            raise RuntimeError(
                "action_record not found. Make sure to record actions during step()."
            )

        T = getattr(self, "max_timesteps", len(self.action_record))
        dt = getattr(self, "policy_dt", 0.02)
        timesteps = np.arange(T) * dt

        a_rec = np.asarray(self.action_record)
        a_dim = getattr(self, "a_dim", a_rec.shape[1])
        n_rec = min(T, len(a_rec))

        # Create dataframe
        data = {"time[s]": timesteps[:n_rec]}
        for i in range(a_dim):
            data[f"action_{i}"] = a_rec[:n_rec, i]
        df = pd.DataFrame(data)

        # Compute RMSE for each action (w.r.t. 0 as baseline)
        rmse = np.sqrt(np.mean(a_rec**2, axis=0))
        rmse_row = {"time[s]": "RMSE"}
        for i in range(a_dim):
            rmse_row[f"action_{i}"] = rmse[i]
        df = pd.concat([df, pd.DataFrame([rmse_row])], ignore_index=True)

        os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(
            save_path
        ) else None
        df.to_csv(save_path, index=False)
        print(f"[Saved] Action data + RMSE metrics -> {save_path}")
