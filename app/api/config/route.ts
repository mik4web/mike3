import type { KnowledgeBase } from "@/types/knowledge-types"

// In-memory storage for configuration (in a real app, you'd use a database)
const config = {
 systemPrompt: `You are a helpful AI assistant called Andrew working for Mike Web Agency that provides support for social media managers and Virtual Assistants. 
You have access to a knowledge base that contains information about social media management tasks, procedures, and policies.

Key behaviors:
1. When provided with relevant context from the knowledge base, use it to answer questions accurately.
2. If no relevant context is provided, use your general knowledge to provide helpful responses.
3. Be concise in all responses. Do not give lengthy answers unless necessary or requested by the question.
4. If a question requires specific company knowledge not provided in the knowledge base, such as financial or personal matters, instruct the user to reach their supervisor Mary on Discord.
5.You have to thank the employee when they share their daily report and help them out how can they improve in their follow-back ratio if that is less than 10% on a given day .
You can check and verify how many  follows employees had to make on a specific day by going to the knowledgeBase section with   id: 'daily-activities' and then calculating if the employee achieved the 10% goal on their provided day . 
`,

  // Pre-chunked knowledge base
  knowledgeBase: [
    {
      id: "creating-multiple-accounts",
      title: "Creating Multiple Accounts (IG, Twitter, TikTok)",
      content: `1. Creating Multiple Accounts (IG, Twitter, TikTok)
a. Use a Mobile Device
You must create and operate these accounts using a mobile device. Many essential features are only accessible through mobile apps, making this a crucial step.

b. Account Verification
To verify your accounts, follow these steps:
    1. Email Verification
        - You can use existing emails or create new ones.
        - A total of three email accounts are sufficient for verifying all platforms.
    2. Phone Number Verification
        - SMS verification is crucial to prevent future restrictions and CAPTCHA challenges.
        - You can use any phone number to receive the verification code.
        - If verifying all accounts by phone isn't possible, ensure at least the first account on each platform is verified with both email and phone.
        - Platforms like Twitter often allow multiple accounts to be verified with the same phone number.

Social media platforms require verification to confirm the accounts are managed by a human rather than an automated system.
Optional: If you have an extra mobile device, consider spreading the accounts across different phones to diversify operations.

!!! IMPORTANT NOTE
If Instagram or Twitter restricts you from creating multiple accounts in one day, this means they have a limit in place. In this case, wait and create the remaining accounts over the following days until you reach three accounts per platform.

2. Choosing a Username & Name
    - Usernames and display names should be of English origin.
    - You can use this tool or similar tools to generate names: https://www.namegeneratorfun.com/american
    - The same names can be reused across different platforms.

3. Handling Account Bans
    - If an account gets banned, click the appeal button—most accounts are reinstated within a day.
    - If the appeal is unsuccessful, simply create a new account.
    - If multiple newly created accounts get banned repeatedly, continue working with the remaining accounts.

!!! CRITICAL WARNING
Do NOT follow the main Mike Web profile (@mike_dev_defi) with any of your created or personal accounts.
Doing so makes it easier for Instagram, TikTok, and Twitter to link your marketing accounts to our main profile, identifying them as marketing accounts. This will:
    ✅ Trigger severe limitations on new accounts
    ✅ Increase CAPTCHA challenges
    ✅ Make account management extremely difficult

4. Adjusting Privacy Settings
After creating the accounts, change Direct Message (DM) settings to allow messages from everyone.
    - Unlike Instagram, TikTok and Twitter do not have a DM request system, meaning users can't message you unless your settings are adjusted.`,
      keywords: ["accounts", "verification", "mobile", "username", "account bans", "privacy settings"],
      relatedChunks: ["profile-picture-bio", "posting-guidelines", "dm-strategy"],
    },
    {
      id: "profile-picture-bio",
      title: "Profile Picture & Bio",
      content: `2. Profile Picture & Bio
The profile picture and bio will be provided via Discord after you successfully create accounts on all mentioned platforms.`,
      keywords: ["profile picture", "bio", "discord", "accounts"],
      relatedChunks: ["creating-multiple-accounts", "posting-guidelines"],
    },
    {
      id: "posting-guidelines",
      title: "Posting Guidelines",
      content: `3. Posting Guidelines

Initial Posts
Each newly created account must have 9 initial posts to appear legitimate and active.
    - These posts can be uploaded on the same day.
    - Images and videos for these posts will be provided via Discord.
    
⚠ Video Download Warning:
Do not use Instagram's built-in download tool, as it adds a copyright logo.
Instead, use third-party tools like:
🔗 https://snapinsta.app/instagram-reels-video-download or any other tool/app that you like

Important Notice for Twitter (X):
Twitter (X) may display a paywall requiring a $1 annual fee to post on their platform.
🚀 Bypass Method:
    1. Go to Twitter Profile Settings 
    2. Change your country to the USA
This should remove the paywall restriction.

Content Usage Rules:
    ✅ Same content can be posted across different platforms (e.g., Instagram 1, TikTok 1, Twitter 1).
    ❌ Different content must be used within the same platform (e.g., Instagram 1 ≠ Instagram 2).
This helps prevent account bans.

Regular/Daily Posts
After the initial setup, each account should post once daily to stay active.
    - Captions: 
        • You can write anything relevant to the video or leave it blank.
    - Hashtags: 
        • Use 6-7 hashtags per post to maximize reach.
        • Suggested hashtags:
        
#nftartwork #opensea #nftcommunity #3dart #metaverse #crypto #memecoin #cryptoart #ai #midjourney #aicommunity #blender3d #surrealism #3dartwork #aiart`,
      keywords: ["posting", "initial posts", "daily posts", "hashtags", "content usage", "social media"],
      relatedChunks: ["creating-multiple-accounts", "daily-activities"],
    },
    {
      id: "daily-activities",
      title: "Daily Activities",
      content: `3. Daily Activities
Gradual Activity Build-Up
As a Social Media Manager, you must be aware that new accounts come with strict limitations on following, liking, and engaging with content. To avoid getting flagged, we follow a 10-day warm-up period to gradually increase activity levels and maximize engagement without triggering platform restrictions.
Each platform (Instagram, TikTok, Twitter) has daily activity limits to prevent spam, especially for new accounts.
⚠️ Strictly follow the table below to stay within safe limits.

📊 Table of Daily Actions (Per Account)
Day
Follows (Twitter & TikTok)
Likes (All Platforms)
Follows (Instagram)
Day 1: ~60, 20-30, ~40
Day 2: ~80, 30-40, ~40
Day 3: ~80, 30-40, ~40
Day 4: ~100, 30-40, ~60
Day 5: ~100, 40-50, ~60
Day 6: ~120, 40-50, ~80
Day 7: ~140, 40-50, ~80
Day 8: ~160, 40-50, ~80
Day 9: ~200, 40-50, ~100
Day 10: ~230, 40-50, ~120

✅ After Day 10 (Full Potential Reached):
    • Twitter/TikTok: ~200-230 follows per day per account 
    • Instagram: ~120 follows per day per account 
    • All Platforms: ~50 likes per day per account 

📌 Why Instagram Has Lower Limits?
Instagram is more sensitive to daily follows, so the numbers are lower than Twitter/TikTok to prevent bans.

⚠️ Social Media Activity Limits You Must Follow
🚀 Following Limits (Per Hour & Day)
⚠ New accounts have strict follow limits for the first 10-15 days.
    • Max follows per hour: 
        ◦ Instagram: 20-30 follows 
        ◦ TikTok/Twitter: 40-50 follows 
    • Exceeding these limits can lead to restrictions or account bans. 

💡 Pro Tip: Rotate Activity Between Platforms
Instead of doing all actions at once, spread them out throughout the day to avoid being flagged as spam.

✅ Example Work Schedule (Activity Rotation)
🔹 At 10:00 AM
    1. 40 actions on Instagram Account 1 
    2. 40 actions on Twitter Account 1 
    3. 40 actions on TikTok Account 1 
    4. 40 actions on Instagram Account 2 
    5. 40 actions on Twitter Account 2 
    6. 40 actions on TikTok Account 2 
🔹 At 12:00 PM → Repeat the same steps
🔹 At 2:00 PM → Repeat again

This method allows you to complete all required daily follows/likes without hitting limits or triggering spam filters.
📌 Feel free to adjust this schedule based on your availability.

❤️ Liking Strategy (Why It's Important)
Liking posts daily helps attract new followers.
    • Users see your likes, check your profile, and may follow you back. 
    • Just like follows, likes also have limits, so follow the daily quota to prevent bans. 

🚨 What To Do If You Get Temporarily Blocked?
If Instagram/Twitter/TikTok shows a "temporary suspension" message:
    ✅ Reduce follows/likes by 30% for 2-3 days until normal activity resumes.
If your account is suspended and you can't log in:
    ✅ Look for the "APPEAL" button and submit a request.
    ✅ Most accounts get restored within 24 hours after appealing.`,
      keywords: [
        "daily activities",
        "activity limits",
        "warm-up period",
        "social media manager",
        "activity rotation",
        "liking strategy",
        "temporary block",
      ],
      relatedChunks: ["follow-like-strategy", "reporting-assistance", "dm-strategy"],
    },
    {
      id: "follow-like-strategy",
      title: "How & Where to Follow and Like",
      content: `4. How & Where to Follow
📌 Targeted Following Strategy.
There is a 2 STEP process in following people on a daily basis.

📌 STEP 1:
1️⃣ You will follow only through target references selected by our algorithms and given to you on Discord chat.
    • You cannot follow random accounts.
    • The selected targets we have chosen have the highest conversion rate for leads and clients.
2️⃣ Your target profiles will be provided once your accounts are set up and have initial posts.
3️⃣ At the end of the month, we will require proof that you have followed the correct targets.

🛠️ How to Follow the Target Accounts

Recently social media platforms (IG, X, TikTok) have disabled the ability to see someone's full "Follower" list tab. Now you can currently view only around 50 people, which has made it extremely difficult to mass follow on a daily basis. This is why we have implemented a clever strategy on generating mass following from a reference (target) point.

📌 Step-by-Step Following Method:
1️⃣ Go to your assigned target profile provided by us on Discord.
2️⃣ Click on its "Followers" tab.
3️⃣ Open the profile of the 1st follower and follow their audience (followers).
4️⃣ Once finished, move to the 2nd follower's profile and follow their audience.
5️⃣ Repeat for the 3rd, 4th, … up to the 50th+ follower of the target account.
💡 This ensures you always have fresh accounts to follow without limitations.

Here is an image illustration to make things clearer 

📌 STEP 2: What to Do After Running Out of People to Follow on the provided targets?

✅ By the time you run out of accounts to follow using the STEP 1 method, your own "Followers" list will have already grown significantly due to follow-backs.
👉 You can now simply start following the audience of the people that are following you back.
🛠️ Steps to Follow New Accounts from Your "Followers" List:
1️⃣ Go to your account profile.
2️⃣ Open the "Followers" tab.
3️⃣ Start following the followers of each person on that list.
💡 So the idea is simple: you use your new friends' audience as targets to follow new people. This ensures a continuous flow of new accounts to follow.

CAUTION !! VERY IMPORTANT
🚨 Do not follow the same people twice.
How to Avoid Following the Same People Twice when daily following:
✅ Double-check the target profile before following its followers. Always verify if you have already used this profile as a source.
✅ Keep a record of the last profile you used to gather followers. This helps you avoid repeating the same target.

It is crucial not to follow the same people twice since it would defeat the purpose of the job.

🚨 Each of your accounts will have ONLY ONE initial assigned target (reference).
Example: Account 1 corresponds to target reference 1,
         Account 2 corresponds to target reference 2,
         Account 3 corresponds to target reference 3.
Do not mix these target references to prevent following the same people.

📊 Follow-Back Ratio
📌 Mass following typically results in a 10% follow-back rate.
✅ For every 100 follows, expect at least 10 people to follow back.
⚠️ If you're not achieving this rate, notify us immediately—it may indicate the target reference has too many bot accounts.

🛑 How to Avoid Bot Accounts?
⚠️ Social media platforms—especially in our niche—are full of bots and fake accounts.
Here are key signs to watch out for when following or liking accounts:
🚩 Red Flags for Bot Accounts:
❌ Avoid accounts with Indian/Arabic-looking names.
    • These audiences are often filled with bots and are not our target.
❌ Avoid following accounts with seductive (half-naked) female profile pictures.
    • These are usually fake bot accounts.
❌ Beware of strange, autogenerated usernames.
    • Example: user19239921893, user293128329819.
    • If many accounts have usernames like this, they are likely bots.
❌ Check the follow-back ratio.
    • For every 100 follows, you should get at least 10 follow-backs.
    • If you're getting significantly less, notify us immediately—your assigned target may contain too many bot accounts.

📌 6. How & Where to LIKE
🛠️ The Liking Strategy
Liking posts is just as important as following—it increases engagement and visibility.
To maximize effectiveness, we will use #hashtags to find the best posts.
These #hashtags will be provided in Discord as well.

📌 Step-by-Step Guide to Liking on Each Platform
✅ 1. Use the Provided Hashtag Targets
    • Instagram: Go to the hashtag page and select the "Recent" filter.
    • Twitter (X): Search the hashtag and select the "Latest" filter.
    • TikTok: Search the hashtag, then use filters to show posts uploaded in the last month or select "Recently Uploaded."
✅ 2. Find Posts with High Engagement
    • Look for posts with many comments.
    • Prioritize comments that have low likes (0-10 likes).
    • Liking these comments increases the chances of users noticing you.
✅ 3. Alternative Method: Liking Low-Engagement Posts
    • Instead of comments, you can like posts within the provided hashtags.
    • Focus on posts with 0-20 likes to ensure maximum visibility.
👉 You can choose either method—liking comments or low-like posts—to hit your daily goal!`,
      keywords: ["follow", "target", "strategy", "social media", "bot detection", "like", "hashtag", "engagement"],
      relatedChunks: ["daily-activities", "dm-strategy"],
    },
    {
      id: "dm-strategy",
      title: "DM Strategy and Message Templates",
      content: `7- DM's  The most crucial part.

Hopefully you understood the following and liking process and I didn't over-complicate those simple steps.

Now we need to understand what to do with our new gained audience.

1- First of all, make sure that you are following all your followers (mutually following each other) otherwise you won't be able to send messages.

The purpose of DMs will be to drive traffic, audience, and potential customers to our company.

There is a 3 STEP process in which you will engage with your new audience:

📌 STEP 1:

HERE ARE THE MESSAGE TEMPLATES ==>
Send them one of the template messages we've provided. We've included multiple variations to prevent social media platforms from flagging the messages as spam, so be sure to switch between them regularly to stay under the radar.

Version 1:
Hey! Really appreciate the follow 😊 I’m with Mike Web3 Development team . We love building cool web projects—no upfront fees. If you’re curious, we can:
1️⃣ Craft your dream website—for free , pay only if you love it
2️⃣ Launch your own meme-coin free —your idea, our  marketing/development
3️⃣ Code custom bots & automations—free build, pay when approved
Got something in mind? Just DM us at the main profile   @mike_dev_defi!



Version 2:

Hi there! Thanks for connecting 😊 I’m part of the Mike Web3 development team. We build anything on the internet —zero cost until you love it. Here’s what we offer:
1️⃣ Futuristic website—free design, pay only if you like it
2️⃣ Meme-coin creation—your concept + our promo and development
3️⃣ Trading bots & other AI  automations—built free, pay on sign-off
Feel free to ping us at the main profile  @mike_dev_defi anytime!




Version 3:

Hello and thanks for the follow 😊 I’m with Mike Web3 Development. We bring your web ideas to life—no payment until you’re satisfied. We can:
1️⃣ Design your perfect website—for free, you pay only on approval
2️⃣ Set up a profitable meme-coin—your vision, our marketing/development
3️⃣ Develop trading bots & AI automations—free build, pay later
If any of this sounds good, shoot a DM to our main account  @mike_dev_defi!

Version 4:

Hi there! Thanks for following  😊 I’m with Mike Web3 Development company. We build anything on the web—no upfront cost, you pay only if you love it.
1️⃣ Dream website built for free—pay after you’re happy
2️⃣ Profitable web3 business like  meme-coin setup—your idea + our marketing/development 
3️⃣ Custom trading bot or AI automation—built free, pay on approval 
Let me know if you have anything in mind and  DM it at our main profile  @mike_dev_defi!

📌 STEP 2:
⚠️ TikTok does not recognize the @username command, meaning users won't be redirected to the profile when mentioned in DMs.
🛠️ Solution: Send a Pinned Video for Higher Visibility.
Immediately after sending the initial STEP 1 text message, send the 1st pinned video from @mike_dev_defi to the person you're DM-ing.
📌 Steps to Share the Pinned Video:
1️⃣ Go to @mike_dev_defi on TikTok.
2️⃣ Open the 1st pinned video.
3️⃣ Click "Share" → Select "Send to Friends" → Choose the person you just DM-ed.
✅ Why is this important?
    • Ensures the recipient sees the video directly in their inbox.
    • Increases the chances of engagement and response.
🚫 Skip this step for Instagram and Twitter (X), as they recognize @username mentions.

📌 STEP 3: Responding to Inquiries
If someone asks for more details about the DM proposal, redirect them to our main page using this pre-written response:
📩 "Please DM Mike Web (@mike_dev_defi) for further discussion."
This keeps communication streamlined and ensures potential leads are handled properly.`,
      keywords: ["dm", "message templates", "social media", "pinned video", "inquiries"],
      relatedChunks: ["daily-activities", "reporting-assistance"],
    },
    {
      id: "reporting-assistance",
      title: "Technical Assistance & Reporting",
      content: `📌 STEP 8: Technical/General Assistance & Reporting

1️⃣ AI Chatbot for Technical & Work-Related Questions
To help you navigate day-to-day technical and general questions—especially during your first month—we have set up our own AI Chatbot, specifically trained for your scope of work.
💬 Access your AI Assistant here: https://chat.mike3web.com/
📌 Use this chatbot for:
    ✔️ Troubleshooting issues with your accounts.
    ✔️ Guidance on operational processes.
    ✔️ Any work-related technical inquiries.

2️⃣ Contact Your Supervisor for Personal & Financial Matters
For any personal or financial concerns, please reach out to your supervisor:
    👤 Supervisor: Mary
    💬 Contact: Discord

3️⃣📊 Regular reporting is required as proof of your work.
A report of your daily activities must be submitted every day to your AI Chatbot.
📌 Sample Report Format:
Platform: Instagram/Twitter/TikTok
📅 Day 1:
Metric
Acc 1, Acc 2, Acc 3
Follow backs: X1, X2, X3
Leads (Responses): Y1, Y2, Y3
📅 Day 2, 3, 4, etc.: Same format as Day 1.
➡ Where:
    • X1, X2, X3 = Number of follow-backs  you gain per  account.
    • Y1, Y2, Y3 = Number of replies, questions you get from DM's that you send  
📌 How to Submit the Report?
    • Simply copy-paste the report directly into the AI assistant chat box.
         

    
📌 STEP 9: Getting Started
🎯 Good luck on your journey with us!
📌 Next Steps:
1️⃣ Create the required accounts and reach out to your supervisor on Discord to receive the relevant content to start work.`,
      keywords: ["reporting", "technical assistance", "AI chatbot", "supervisor", "daily report", "getting started"],
      relatedChunks: ["daily-activities", "dm-strategy"],
    },
    // Add more chunks as needed
  ] as KnowledgeBase,
}

export async function GET() {
  return Response.json({
    success: true,
    systemPrompt: config.systemPrompt,
    knowledgeBase: config.knowledgeBase,
  })
}

export async function POST(req: Request) {
  try {
    const data = await req.json()

    // Update configuration
    if (data.systemPrompt !== undefined) {
      config.systemPrompt = data.systemPrompt
    }

    if (data.knowledgeBase !== undefined) {
      config.knowledgeBase = data.knowledgeBase
    }

    return Response.json({
      success: true,
      message: "Configuration updated successfully",
    })
  } catch (error) {
    console.error("Config update error:", error)
    return Response.json({ success: false, error: "Failed to update configuration" }, { status: 500 })
  }
}

