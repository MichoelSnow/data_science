# gcloud wide flags

These flags are available to all commands: `--account`, `--configuration`, `--flatten`, `--format`, `--help`, `--log-http`, `--project`, `--quiet`, `--trace-token`, `--user-output-enabled`, `--verbosity`. Run `gcloud help` for details.

Not all options are listed just ones I most commonly need to use

# gcloud auth

## gcloud auth list

```
gcloud auth list
```

Lists accounts whose credentials have been obtained and shows which account is active

## gcloud auth login

```
gcloud auth login [ACCOUNT] [Options]
```


Obtains access credentials for your user account via a web-based authorization flow. When this command completes successfully, it sets the active account in the current configuration to the account specified.

### Example

```
gcloud auth login myemail@gmail.com
```

| Name, shorthand       | Default       | Description   |
| --------------------- |---------------| -------------|
| `--activate`          | Enabled       | Set the new account to active, use `--no-activate` to disable        |
| `--launch-browser`    | Enabled       | Launch a browser for authorization. If not enabled or DISPLAY variable is not set, prints a URL to standard output to be copied. Use `--no-launch-browser` to disable.      |


## gcloud auth revoke

```
gcloud auth revoke [ACCOUNTS]
```

Revokes credentials for the specified user accounts or service accounts. When you revoke the credentials, they are removed from the local machine. If no account is specified, this command revokes credentials for the currently active account, effectively logging out of that account.


# gcloud compute

## gcloud compute instances list

```
gcloud compute instances list
```

displays all Google Compute Engine instances in a project.


## gcloud compute instances start

```
gcloud compute instances start INSTANCE_NAMES
```

start a stopped Google Compute Engine virtual machine. Only a stopped virtual machine can be started.


## gcloud compute instances stop

```
gcloud compute instances start INSTANCE_NAMES
```

stop a running Google Compute Engine virtual machine. Stopping a VM performs a clean shutdown, much like invoking the shutdown functionality of a workstation or laptop. Stopping a VM with a local SSD is not supported and will result in an API error.


## gcloud compute ssh

```
gcloud compute ssh [USER@]INSTANCE [Options]
```

A thin wrapper around the `ssh(1)` command that takes care of authentication and the translation of the instance name into an IP address.  Note, this command does not work for Windows VMs.

| Name, shorthand       | Default       | Description   |
| --------------------- |---------------| -------------|
| `--ssh-flag=SSH_FLAG` |               | Additional flags to be passed to `ssh(1)`. It is recommended that flags be passed using an assignment operator and quotes.        |
| `--ssh-key-file=SSH_KEY_FILE`    | ~/.ssh/google_compute_engine       | path to the SSH key file     |


### Example

```
gcloud compute ssh msnow@instance-1 --ssh-flag="-L 8898:localhost:8898"
```

# gcloud config

## gcloud config set

```
gcloud config set SECTION/PROPERTY VALUE
```

sets the specified property in your active configuration only.   To view a list of properties currently in use, run `gcloud config list`

Note that SECTION/ is optional while referring to properties in the core section, i.e., using either core/project or project is a valid way of setting a project, while using section names is essential for setting specific properties like compute/region.



| SECTION/PROPERTY      | Default       | Description   |
| --------------------- |---------------| -------------|
| `core/account` |               | Account gcloud should use for authentication|
| `core/project` |               | Project ID (not project name) of the Cloud Platform project to operate on by default.|


### Example

```
gcloud config set core/account your-email-account@gmail.com
gcloud config set account your-email-account@gmail.com
```
set an existing account to be the current active account

```
gcloud config set compute/zone asia-east1-b
```

set the zone property in the compute section

# gcloud init

```
gcloud init [--console-only]
```
gcloud init launches an interactive Getting Started workflow for gcloud. It performs the following setup steps:
- Authorizes gcloud and other SDK tools to access Google Cloud Platform using your user account credentials, or lets you select from accounts whose credentials are already available.
- Sets properties in a gcloud configuration, including the current project and the default Google Compute Engine region and zone.

use the `--console-only` flag to prevent the command from launching a browser for authiroization.  Instead it will ask you to go to the site and copy the necessary key.


# gcloud projects

## gcloud projects list

```
gcloud projects list
```

Lists all active projects, where the active account has Owner, Editor or Viewer permissions.