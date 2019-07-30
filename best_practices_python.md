Best practices for coding in python culled from personal experience and other people's best practices lists



# Functions/Methods

Functions should be less than a page of code.  They should be quick to understand by people unfamiliar with the code

## General

- variables and other infromation should be introduced as close as possible to where they are used, e.g., declaring an empty list right before filling it
- Use built-in exceptions over custom one wherever appropriate


## Structure

Functions are made up of an input a transform and an output, in that order

- inputs
  - Set up expectations for the reader, i.e., the first few lines should give an idea of where the function is going
  - Gather all the information you need and throw out everything you do not need
    - Early errors are good errors, i.e., if you do not have the information you need, abort with a nicely described error
    - Asserts add information
  - Return early to add confidence, e.g., give the option to return part of the output early (or maybe all of it in logging) so that users can see what the code will do before they implement it for real (dry run)
    ```
    def delete_all_uers(dry_run=False):
        users = get_all_users()

        if dry_run:
            logging.info('dry-run, not delecting %d users' % len(users))
            return
    ```
- transform
  - This should follow logically from your input and function name, i.e., there should be no surprises in the code
  - Exceptions at this point should be exceptional
- output
  - Format and return your information in the way that the caller expects
    - Return explicitly and consistently, i.e., try to avoid implicit `return None` and try to have returns be of same type, even if it is an empty version of that type
    - Make sure to think about code failures when creating returns
  - Exceptions at this point should be really really surprising


