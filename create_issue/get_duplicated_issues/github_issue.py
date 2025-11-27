import re
import requests
from github import Github
from github import Auth
from github.GithubException import GithubException
import time
import os
import zipfile
from urllib.parse import urlparse


class Github_Issue:
    def __init__(
        self,
        repo,
        token,

    ):
        # using an access token
        auth = Auth.Token(token)

        # Public Web Github
        g = Github(auth=auth)

        # Replace with the repository owner and name
        self.repo = g.get_repo(repo)
        self.token = token
        self.repo_name = repo


    def get_issues(self, state):
        # Get all open issues, excluding pull requests
        issues = self.repo.get_issues(state=state)
        return issues

    def get_issue(self, ticket):
        if ticket is not None and isinstance(ticket, int):
            issue = self.repo.get_issue(number=ticket)
            return issue
        else:
            raise AssertionError()
            return None

    def get_comments(self, ticket):

        if ticket is not None and isinstance(ticket, int):

            # Get the issue
            issue = self.repo.get_issue(number=ticket)

            # Get the comments
            comments = issue.get_comments()

            comments_contents = "" 

            # process the comments
            for comment in comments:
                # Do not support url parsing at present
                if comment.body != None and comment.body != "" and comment.body != "[](url)":
                    comments_contents = comments_contents + "{"
                    comments_contents = comments_contents + f"Author: {comment.user.login}, "
                    comments_contents = comments_contents + f" Date: {comment.created_at}, "
                    comments_contents = comments_contents + f" Comment: {comment.body}\n"
                    comments_contents = comments_contents + "},"

            return comments_contents
        else:
            raise AssertionError()
            return None


    def get_workflow_runs_for_pr(self, pr_number, token):
        """Get workflow runs for a specific pull request"""
        url = f"https://api.github.com/repos/{self.repo_name}/actions/runs?event=pull_request&per_page=100"
        headers = {
            'Authorization': f'token {self.token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            runs = response.json()['workflow_runs']
            return [run for run in runs if 'pull_requests' in run and len(run['pull_requests']) != 0 and run['pull_requests'][0]['number'] == pr_number]
        except Exception as e:
            print(f"Error fetching workflow runs for PR #{pr_number}: {str(e)}")
            return []
    
    def get_artifact(self, run_id, artifact_name, token):
        """Get artifact details for a specific workflow run"""
        url = f"https://api.github.com/repos/{self.repo_name}/actions/runs/{run_id}/artifacts"
        headers = {
            'Authorization': f'token {token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            artifacts = response.json()['artifacts']
            return next((a for a in artifacts if a['name'] == artifact_name), None)
        except Exception as e:
            print(f"Error fetching artifacts for run {run_id}: {str(e)}")
            return None
    
    def download_and_extract_artifact(self, artifact_id, artifact_name, pr_number, run_id, token, output_dir):
        """Download and extract a specific artifact"""
        url = f"https://api.github.com/repos/{self.repo_name}/actions/artifacts/{artifact_id}/zip"
        headers = {
            'Authorization': f'token {token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        
        pr_dir = os.path.join(output_dir, f"PR-{pr_number}")
        os.makedirs(pr_dir, exist_ok=True)
        zip_path = os.path.join(pr_dir, f"{artifact_name}-run-{run_id}.zip")
        extract_path = os.path.join(pr_dir, f"{artifact_name}-run-{run_id}")
        
        try:
            print(f"Downloading {artifact_name} for PR #{pr_number}, run {run_id}...")
            
            # Download the zip file
            with requests.get(url, headers=headers, stream=True) as response:
                response.raise_for_status()
                with open(zip_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            
            # Extract the zip file
            print(f"Extracting {artifact_name} for PR #{pr_number}, run {run_id}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            
            # Remove the zip file
            os.remove(zip_path)
            
            print(f"Successfully processed {artifact_name} for PR #{pr_number}, run {run_id}")
            return True
        except Exception as e:
            print(f"Error processing artifact for PR #{pr_number}, run {run_id}: {str(e)}")
            if os.path.exists(zip_path):
                os.remove(zip_path)
            return False


    def add_pr_comment(self, body, pr_number):
        pull = self.repo.get_pull(pr_number)
        pull.create_issue_comment(body=body) 

    def add_comment(self, body, issue_number):
        issue = self.repo.get_issue(issue_number)
        issue.create_comment(body=body)

    def parse_github_issue_attachment(self, content, output_dir):
        """
        Download attachment from GitHub issue comment, transform into text if it is a picture and return the merged comment. 
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Process issue body (first comment)
        if content != None and content != "":
            content = self._process_content_with_attachments(content, output_dir)
 
        print(f"Download complete. Files saved in {output_dir}/")

        return content


    def _process_content_with_attachments(self, text, output_dir):
        """
        Helper function to find and download attachments in comment text.
        """
        if not text:
            return
        result = []
        # Find all markdown image/attachment links
        lines = text.split('\n')
        for line in lines:
            if '](' in line and "recent examples" not in line:  # Simple check for markdown links
                start = line.find('](') + 2
                end = line.find(')', start)
                if start != -1 and end != -1:
                    url = line[start:end]
                    if self._is_downloadable_url(url):
                        filename = self._download_file(url, output_dir)

                        if filename == "":                            
                            continue

                        import magic
                        def get_file_type(file_path):
                            mime = magic.Magic(mime=True)
                            file_type = mime.from_file(file_path)
                            return file_type

                        filename = f"{output_dir}/{filename}"
                        file_type = get_file_type(filename)
                        if 'image' in file_type:
                            from PIL import Image
                            import pytesseract
                            image = Image.open(filename)
                            text = pytesseract.image_to_string(image)
                            result.append(line[0:start])
                            result.append(f"\n```\n{text}\n```\n")
                            result.append(line[end:-1])
                        if 'text' in file_type:
                            f = open(filename, "r")
                            result.append(line[0:start])
                            text = f.read()
                            result.append(f"\n```\n{text}\n```\n")
                            result.append(line[end:-1])
            else:
                result.append(line)
        return "".join(result)

    def _is_downloadable_url(self, url):
        """
        Check if the URL points to a downloadable file.
        """
        # Common attachment hosts on GitHub
        downloadable_hosts = [
            'github.com',
            'githubusercontent.com',
            's3.amazonaws.com',  # Sometimes used for GitHub uploads
            'user-images.githubusercontent.com',
            'torch-ci.com'
        ]

        parsed = urlparse(url)
        if not parsed.scheme in ('http', 'https'):
            return False

        return any(host in parsed.netloc for host in downloadable_hosts)

    def _download_file(self, url, output_dir):
        """
        Download a file from URL to output directory.
        """
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()

            # Extract filename from URL
            filename = os.path.basename(urlparse(url).path)
            if not filename:
                filename = f"attachment_{hash(url)}.bin"  # fallback name

            filepath = os.path.join(output_dir, filename)

            # Save the file
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            print(f"Downloaded: {filename}")
            return filename
        except Exception as e:
            print(f"Failed to download {url}: {str(e)}")
            return ""

    

    def create_issue_with_label(self, title, body, labels=["skipped"]):
        try:
            # Get repository
            repo = self.repo
            
            # Check if "skipped" label exists, create if it doesn't
            try:
                for label in labels:
                    repo.get_label(label)
            except GithubException:
                # Create the label if it doesn't exist
                repo.create_label(
                    name=label,
                    color="0366d6",  # Blue color
                    description="Skipped items or tasks"
                )
                print(f"Created '{label}' label")
            
            # Create the issue
            issue = repo.create_issue(
                title=title,
                body=body,
                labels=labels
            )
            
            print(f"Issue created successfully!")
            print(f"Title: {issue.title}")
            print(f"URL: {issue.html_url}")
            print(f"Labels: {[label.name for label in issue.labels]}")
            
            return issue.number
            
        except GithubException as e:
            print(f"❌ Error creating issue: {e}")
            return None

