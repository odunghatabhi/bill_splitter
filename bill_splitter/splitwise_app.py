import argparse
import os
from dotenv import load_dotenv

# splitwise.py
from splitwise import Splitwise
from splitwise.expense import Expense
from splitwise.user import ExpenseUser
from splitwise.group import Group

load_dotenv()

CONSUMER_KEY = os.getenv("SPLITWISE_CONSUMER_KEY")
CONSUMER_SECRET = os.getenv("SPLITWISE_CONSUMER_SECRET")


def _require_splitwise_consumer_credentials() -> None:
    if CONSUMER_KEY and CONSUMER_SECRET:
        return
    raise RuntimeError(
        "Missing Splitwise consumer credentials. Set SPLITWISE_CONSUMER_KEY and "
        "SPLITWISE_CONSUMER_SECRET in your environment or .env file."
    )

def get_splitwise_client(
    access_token: str = None,
    access_token_secret: str = None,
    api_key: str = None,
) -> Splitwise:
    """
    Returns a Splitwise client. If access_token is None, client can be used to start OAuth flow.
    """
    _require_splitwise_consumer_credentials()
    if api_key:
        return Splitwise(CONSUMER_KEY, CONSUMER_SECRET, api_key=api_key)
    return Splitwise(CONSUMER_KEY, CONSUMER_SECRET, access_token, access_token_secret)


def get_authorize_url() -> tuple[str, str]:
    """
    Returns (url_to_authorize, temporary_oauth_secret)
    """
    s = get_splitwise_client()
    url, oauth_secret = s.getAuthorizeURL()
    return url, oauth_secret


def get_access_token(oauth_token: str, oauth_secret: str, oauth_verifier: str) -> tuple[str, str]:
    """
    Exchange verifier for permanent access token
    """
    s = get_splitwise_client()
    access_token, access_token_secret = s.getAccessToken(oauth_token, oauth_secret, oauth_verifier)
    return access_token, access_token_secret


def get_groups(access_token: str, access_token_secret: str) -> list[Group]:
    s = get_splitwise_client(access_token, access_token_secret)
    return s.getGroups()


def list_group_names(access_token: str, access_token_secret: str) -> list[tuple[int, str]]:
    groups = get_groups(access_token, access_token_secret)
    return [(g.getId(), g.getName()) for g in groups]


def list_group_names_with_api_key(api_key: str) -> list[tuple[int, str]]:
    """
    API-key-only flow via Splitwise SDK constructor:
    Splitwise(CONSUMER_KEY, CONSUMER_SECRET, api_key=...)
    """
    s = get_splitwise_client(api_key=api_key)
    return [(g.getId(), g.getName()) for g in s.getGroups()]


def get_group_members(access_token: str, access_token_secret: str, group_id: int) -> list[dict]:
    """
    Returns a list of dicts: {id, first_name, last_name, email}
    """
    s = get_splitwise_client(access_token, access_token_secret)
    group = s.getGroup(group_id)
    users = group.getUsers() if hasattr(group, "getUsers") else []
    return [{"id": u.getId(), "name": f"{u.getFirstName()} {u.getLastName()}"} for u in users]


def create_expense(access_token: str, access_token_secret: str, group_id: int, description: str, expense_data: list[dict]):
    """
    expense_data: list of dicts {user_id, paid_share, owed_share}
    """
    s = get_splitwise_client(access_token, access_token_secret)
    expense = Expense()
    expense.setCost(str(sum(float(d["owed_share"]) for d in expense_data)))
    expense.setDescription(description)
    expense.setGroupId(group_id)
    users = []
    for d in expense_data:
        eu = ExpenseUser()
        eu.setId(d["user_id"])
        eu.setPaidShare(str(d.get("paid_share", 0)))
        eu.setOwedShare(str(d["owed_share"]))
        users.append(eu)
    expense.setUsers(users)
    return s.createExpense(expense)


def _main() -> int:
    parser = argparse.ArgumentParser(description="Standalone Splitwise connectivity test")
    parser.add_argument("--api-key", default=os.getenv("SPLITWISE_API_KEY"))
    parser.add_argument("--access-token", default=os.getenv("SPLITWISE_ACCESS_TOKEN"))
    parser.add_argument("--access-token-secret", default=os.getenv("SPLITWISE_ACCESS_TOKEN_SECRET", ""))
    args = parser.parse_args()

    try:
        if args.api_key:
            groups = list_group_names_with_api_key(args.api_key)
        elif args.access_token:
            groups = list_group_names(args.access_token, args.access_token_secret)
        else:
            print(
                "Missing credentials. Provide either:\n"
                "- --api-key (or SPLITWISE_API_KEY), or\n"
                "- --access-token (and optionally --access-token-secret)."
            )
            return 1
    except Exception as exc:
        print(f"Splitwise connection failed: {exc}")
        return 2

    if not groups:
        print("Connected successfully. No groups found.")
        return 0

    print("Connected successfully. Groups:")
    for group_id, group_name in groups:
        print(f"- {group_id}: {group_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
