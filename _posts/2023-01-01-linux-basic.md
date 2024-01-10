---
title: '[Note] Linux basic'
date: 2023-01-01
permalink: /posts/2023/01/01/linux-basic/
tags:
  - research
  - proposal
  - linux
--- 

Linux basic command
======

Here are some of the most basic Linux commands and their explanations:

1. **ls:** Lists files and directories in the current directory.

   ```bash
   ls
   ```

   Output:

   ```
   Desktop Downloads Documents Music Pictures Public Templates Videos
   ```

2. **pwd:** Prints the full path of the current working directory.

   ```bash
   pwd
   ```

   Output:

   ```
   /home/bard
   ```

3. **cd:** Changes the current working directory.

   ```bash
   cd Documents
   ```

   Now, the current directory is `Documents`.

4. **mkdir:** Creates a new directory.

   ```bash
   mkdir new_directory
   ```

   This command creates a directory named `new_directory` in the current directory.

5. **rmdir:** Removes an empty directory.

   ```bash
   rmdir new_directory
   ```

   This command removes the directory `new_directory` if it is empty.

6. **cp:** Copies files or directories.

   ```bash
   cp file1.txt file2.txt
   ```

   This command copies the file `file1.txt` to `file2.txt`.

7. **mv:** Moves or renames files or directories.

   ```bash
   mv file2.txt new_file.txt
   ```

   This command renames the file `file2.txt` to `new_file.txt`.

8. **rm:** Removes files or directories.

   ```bash
   rm old_file.txt
   ```

   This command removes the file `old_file.txt`.

9. **touch:** Creates an empty file.

   ```bash
   touch new_file.txt
   ```

   This command creates an empty file named `new_file.txt` in the current directory.

10. **uname:** Prints information about the Linux system.

   ```bash
   uname -a
   ```

   Output:

   ```
   Linux bard 4.19.152-linuxkit x86_64 GNU/Linux
   ```

These are just a few of the many basic Linux commands that you can use to interact with your system. With a little practice, you'll be able to navigate the Linux command line like a pro.


Shell programming
======

Shell programming is a powerful tool for automating tasks and managing complex systems in Linux. It involves writing scripts that contain a series of commands and instructions, which are then executed by the shell. Shell scripts can be used to perform a wide variety of tasks, such as:

* **Automating repetitive tasks:** Shell scripts can be used to automate tasks that you would otherwise have to perform manually, such as backing up files, configuring system settings, or deploying software.
* **Scripting complex workflows:** Shell scripts can be used to orchestrate complex workflows that involve multiple steps, such as building and deploying an application, or creating a backup and restore process.
* **Integration with other tools:** Shell scripts can be used to integrate with other tools and libraries, such as Python, Perl, or Java. This allows you to create powerful and flexible scripts that can be tailored to your specific needs.

The most common shell used for Linux shell programming is the Bash shell (Bourne-Again Shell). However, there are other shells available, such as the Z shell (Zsh), the Korn shell (ksh), and the TCSH shell (tcsh). Each shell has its own unique syntax and features, so you may want to experiment with different shells to find one that you prefer.

Here are some of the basic elements of shell programming:

* **Variables:** Shell scripts can use variables to store data. Variables are assigned values using the `=` operator. For example, the following line assigns the value "Hello, world!" to the variable `greeting`:

```bash
greeting="Hello, world!"
```

* **Commands:** Shell scripts can execute commands using the `exec` command. For example, the following line executes the `ls` command to list the files in the current directory:

```bash
exec ls
```

* **Input/output redirection:** Shell scripts can redirect input and output from commands. This can be used to capture the output of a command, or to pipe the output of one command to the input of another command. For example, the following line redirects the output of the `date` command to the file `today.txt`:

```bash
date > today.txt
```

* **Control structures:** Shell scripts can use control structures to control the flow of execution. These structures include loops (for, while) and conditional statements (if, else). For example, the following loop prints the numbers from 1 to 10:

```bash
for i in {1..10}; do
  echo $i
done
```

* **Functions:** Shell scripts can define functions to encapsulate reusable code. Functions are defined using the `function` keyword. For example, the following function takes a name as input and prints a greeting:

```bash
function greet {
  echo "Hello, $1!"
}

greet Bard
```

Shell programming is a powerful and versatile tool that can be used to automate tasks, manage systems, and develop custom tools. With a little practice, you can learn the basics of shell programming and start creating your own scripts.


Linux process
======

In Linux, a process is an instance of an executable program that is running on the system. Each process has a unique identifier called a process ID (PID). Processes can be created, deleted, and managed using the shell.

**Creating Processes:**

The most common way to create a process is to execute a command. When you type a command at the shell prompt, the shell creates a child process to execute the command. The child process inherits the environment of the parent process, including the current working directory and the contents of the environment variables.

For example, the following command creates a child process to execute the `ls` command:

```bash
ls
```

This command will list the files in the current directory.

**Starting Processes in the Background:**

By default, processes started from the shell are run in the foreground. This means that the shell will wait for the process to finish before it will allow you to enter another command. However, you can run a process in the background by appending an ampersand (&) to the end of the command.

For example, the following command will start the `top` command in the background:

```bash
top &
```

This command will display the current system resource usage and will continue to run in the background, even though you can enter other commands at the shell prompt.

**Managing Processes:**

The shell provides several commands for managing processes, including:

* `ps`: Displays information about currently running processes
* `kill`: Sends a signal to a process to terminate it
* `pgrep`: Searches for processes by name or PID
* `top`: Displays a real-time view of system resource usage

These commands can be used to monitor the status of processes, kill processes that are not responding, and identify processes that are consuming excessive resources.

**Shell Scripting:**

Shell scripting is a powerful tool for automating tasks involving processes. Shell scripts can be used to create complex workflows that involve multiple processes, handle errors, and manage resource utilization.

By combining the shell's process management capabilities with its scripting language features, you can create powerful tools to automate various tasks and manage your Linux system efficiently.


Job scheduling
======

In Linux, job scheduling refers to the process of automating tasks to run at specific times or intervals. This is a valuable tool for automating repetitive tasks, such as backups, log rotations, and system maintenance. There are several methods for job scheduling in Linux, each with its own strengths and weaknesses.

**Cron:**

Cron is the most popular job scheduling tool in Linux. It is a time-based job scheduler that reads and executes commands from a special file called a crontab. Crontab entries specify the schedule for a job, including the minutes, hours, days of the month, months, and days of the week. Cron jobs can be run under the user's account or with root privileges.

**At:**

At is a one-time job scheduler that allows you to schedule a single job to run at a specified time. At jobs are run in the current user's environment and with the current user's privileges.

**Batch:**

Batch is a job scheduler that is designed to run jobs in the background when the system load is low. Batch jobs are run with the user's privileges and in a special environment that is designed to prevent them from interfering with user-interactive processes.

**Apt-sched:**

Apt-sched is a job scheduler that is specifically designed to schedule package installations and updates. It is integrated with the APT package manager and can be used to schedule automatic package updates at a specified time.

**Comparison of Job Scheduling Tools:**

| Feature | Cron | At | Batch | Apt-sched |
|---|---|---|---|---|
| **Type of job scheduling** | Time-based, recurring | One-time | Background | Package installations/updates |
| **Scheduling granularity** | Minutes, hours, days, months, days of the week | Specific time | Not applicable | Schedules for package installations/updates |
| **Job execution** | Runs jobs in the current user's environment | Runs jobs in the current user's environment | Runs jobs in a special background environment | Runs jobs in a special environment for package installations/updates |
| **Permissions** | Can run jobs under the user's account or with root privileges | Runs jobs under the current user's privileges | Runs jobs with the user's privileges | Runs jobs with the user's privileges |
| **Use cases** | Automation of repetitive tasks | Running a single job at a specific time | Running jobs in the background when the system load is low | Scheduling package installations/updates |

In general, Cron is a good choice for scheduling recurring tasks, At is a good choice for running one-time tasks, Batch is a good choice for running tasks in the background, and Apt-sched is a good choice for scheduling package installations/updates.

**Choosing the Right Job Scheduling Tool:**

The best job scheduling tool for you depends on your specific needs. If you need to automate repetitive tasks that run at specific times or intervals, Cron is a good choice. If you need to run a single job at a specific time, At is a good choice. If you need to run jobs in the background when the system load is low, Batch is a good choice. And if you need to schedule package installations/updates, Apt-sched is a good choice.

Here is a table summarizing the use cases for each job scheduling tool:

| Task | Recommended tool |
|---|
| Automate backups | Cron |
| Log rotations | Cron |
| System maintenance | Cron |
| Running a single job at a specific time | At |
| Running jobs in the background when the system load is low | Batch |
| Scheduling package installations/updates | Apt-sched |



Common Linux Commands Used by Programmers
======

[Link tham khảo]([https://g.co/bard/share/265d74484112](https://ai.plainenglish.io/common-linux-commands-used-by-programmers-62a850156b97))

Let explore with different sections in Linux which is helps programmer to access necessary information from the machine.

File and Directory Operations

✅ls (List):

Usage: ls [options] [directory]

Description: Lists the contents of a directory.

Options:

-l: Long format, displaying detailed information.
-a: Show hidden files.
-h: Human-readable file sizes.

✅cd (Change Directory):

Usage: cd [directory]

Description: Changes the current working directory to the specified one.

✅pwd (Print Working Directory):

Usage: pwd

Description: Prints the current working directory, showing the full path.

✅mkdir (Make Directory):

Usage: mkdir [directory]

Description: Creates a new directory with the specified name.

✅cp (Copy):

Usage: cp [options] source destination

Description: Copies files or directories from the source to the destination.

Options:

-r: Copy directories recursively.
-i: Prompt before overwriting.

✅mv (Move):

Usage: mv [options] source destination

Description: Moves files or directories from the source to the destination, or renames a file.

✅rm (Remove):

Usage: rm [options] file

Description: Removes (deletes) files or directories.

Options:

-r: Remove directories and their contents recursively.
-f: Force, ignore nonexistent files and do not prompt.

✅touch:

Usage: touch [options] file

Description: Creates an empty file or updates the access and modification times of a file.

User Management

The sudo command in Unix-like operating systems is used to execute a command as a superuser or another user, as specified by the security policy configured in the sudoers file. Here's the basic usage of

sudo [OPTION] COMMAND [ARGUMENTS...]

OPTION: Optional flags or options for sudo.

COMMAND: The command you want to execute with elevated privileges.

ARGUMENTS: Any arguments or options required by the specified command.

Usage Examples:

1.Run a Command as Superuser:

sudo ls /root
This command runs the ls command with elevated privileges, allowing it to list the contents of the /root directory, which is typically restricted to the root user.

2. Edit a System Configuration File:

sudo nano /etc/nginx/nginx.conf
This example opens the NGINX configuration file for editing with the Nano text editor. Editing system configuration files usually requires superuser privileges.

3. Install Software:

sudo apt-get install nginx
The sudo command is often used with package management tools, such as apt-get on Debian-based systems, to install or remove software.

4. Restart a System Service:

sudo systemctl restart apache2
Restarting a system service, like Apache, usually requires superuser privileges. sudo allows you to perform such actions.

5. Run a Command as Another User:

sudo -u username command
Use the -u option to run a command as a specified user. Replace username with the desired username.

Text Manipulation

✅cat (Concatenate):

Usage: cat [file]

Description: Displays the content of a file.

✅nano and vim:

Usage: nano [file] or vim [file]

Description: Text editors for creating or editing files. They have different user interfaces and capabilities.

✅grep (Global Regular Expression Print):

Usage: grep [options] pattern [file]

Description: Searches for a pattern in files.

Options:

-i: Ignore case.
-r: Search recursively in directories.

✅sed (Stream Editor):

Usage: sed [options] 's/pattern/replacement/' file

Description: Filters and transforms text using patterns.

Example: sed 's/old/new/' filename

✅awk:

Usage: awk 'pattern { action }' file

Description: A pattern scanning and text processing tool.

System Information

✅top or htop(Display real-time system statistics):

Usage: top

Usage: htop

Description: Both top and htop display real-time system statistics, including information about processes, CPU usage, memory usage, and system resource distribution.

✅free(Display amount of free and used memory in the system):

Usage: free [options]

Description: The free command provides information about the system's memory usage, displaying the total, used, and free memory in kilobytes. Options like -h can be used for human-readable output.

Example: free -h

✅df: Display disk space usage:

Usage :df [options] [file|directory]

Description: The df command shows the disk space usage of file systems. Adding the -h option provides human-readable output.

Example :df -h

Process Management

✅ps: Display information about active processes:

Usage:ps [options]

Description: The ps command provides a snapshot of currently running processes. It displays information such as process ID (PID), terminal associated with the process, CPU and memory usage, and the command that started the process.

Options:

Common options include:
-e: Display information for all processes.
-f: Full-format listing.
-u user: Display processes for a specific user.

✅kill: Send a signal to a process (e.g., terminate a process):

Usage:kill [signal] PID

Description: The kill command sends a signal to a process, allowing for various actions such as terminating, stopping, or reloading. The default signal is SIGTERM, which terminates the process gracefully. Use SIGKILL for forceful termination.

Example:kill 1234 (Terminates the process with PID 1234)

✅pkill: Kill processes based on name:

Usage:pkill [options] pattern

Description: The pkill command sends signals to processes based on their name. It terminates processes that match the specified pattern.

Options: Common options include:

-signal: Specify the signal to send.

-u username: Limit the operation to processes owned by a specific user.

Example: pkill -TERM firefox (Terminates all processes with the name 'firefox')

✅killall: Kill processes by name:

Usage:killall [options] process_name

Description: The killall command sends signals to processes based on their name, similar to pkill. It terminates processes that match the specified name.

Options:

Common options include:

-signal: Specify the signal to send.

-u username: Limit the operation to processes owned by a specific user.

Example: killall -TERM chrome (Terminates all processes with the name 'chrome')

SSH (Secure shell)

✅ssh: Connect to a remote server securely:

Usage:ssh [user@]hostname [options]

Description: The ssh command establishes a secure shell connection to a remote server. It prompts for the user's password or uses key-based authentication. Once connected, users can execute commands on the remote server's shell.

Options:

-p port: Specify the port to connect to (default is 22).
-i identity_file: Specify the private key file for authentication.

Example: ssh user@example.com (Connect to the remote server "example.com" as the user "user")

✅scp: Copy files between a local and remote machine over SSH:

Usage:scp [options] source destination

Description: The scp command securely copies files between a local and a remote machine over an SSH connection. It supports copying to and from remote servers, as well as between two remote servers.

Options:

Common options include:
-P port: Specify the port on the remote server.
-r: Recursively copy entire directories.

Examples:

Copy local file to remote server:

scp localfile.txt user@example.com:/path/to/destination/

Copy from remote server to local machine:

scp user@example.com:/path/to/remotefile.txt /local/destination/

File Compression and Archiving

✅tar: Create and extract tar archives:

Creation of Tar Archive:

Usage:tar -cvf archive.tar [files/directories]

Description: The tar command is used to create tar archives. The options used here are:

-c: Create a new archive.
-v: Verbosely list the files processed.
-f: Use archive file specified (in this case, "archive.tar").
Example:tar -cvf archive.tar file1.txt dir1/

Extraction of Tar Archive:

Usage:tar -xvf archive.tar [files/directories]

Description: The tar command with different options (-x for extract) is used to extract files from a tar archive.

Example: tar -xvf archive.tar

✅gzip (GNU Zip):

Compression:

Usage: gzip [options] file
Description: gzip compresses files and replaces them with a compressed version with a .gz extension.

Example: gzip myfile.txt (Creates myfile.txt.gz)

Decompression:

Usage: gzip -d file.gz or gunzip file.gz

Description: Decompresses a file compressed with gzip.

Example: gunzip myfile.txt.gz (Restores myfile.txt)

These are common commands cover a broad range of Linux functionality and mastering them will help you effectively working on Linux os.


References
======

[Link tham khảo 01](https://g.co/bard/share/265d74484112)

[Link tham khảo 02](https://www.linkedin.com/learning/linux-files-and-permissions-14025387/deleting-files-and-dirs?u=93557497)

[Link tham khảo 03](https://www.linkedin.com/learning/linux-shells-and-processes-14269702/be-more-productive-in-the-linux-shell?u=93557497)

[Link tham khảo 04](https://www.linkedin.com/learning/learning-linux-command-line-14447912/learning-linux-command-line?u=93557497)

Hết.
