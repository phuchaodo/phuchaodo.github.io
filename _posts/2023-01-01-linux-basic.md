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




References
======

[Link tham khảo 01](https://g.co/bard/share/265d74484112)

[Link tham khảo 02](https://www.linkedin.com/learning/linux-files-and-permissions-14025387/deleting-files-and-dirs?u=93557497)

[Link tham khảo 03](https://www.linkedin.com/learning/linux-shells-and-processes-14269702/be-more-productive-in-the-linux-shell?u=93557497)

[Link tham khảo 03](https://www.linkedin.com/learning/learning-linux-command-line-14447912/learning-linux-command-line?u=93557497)

Hết.
